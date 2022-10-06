import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from norm import maxmin_norm
from Similarity import CDist
from Similarity import Buildgraph
from Env import DataEnv
import numpy as np
import copy
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--tau',  default=0.002, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)

parser.add_argument('--learning_rate', default=3e-8, type=float)
parser.add_argument('--gamma', default=0.9, type=int) # discounted factor
parser.add_argument('--capacity', default=80000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=200, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=256, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.0, type=float)
parser.add_argument('--noise_clip', default=0.0, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)


parser.add_argument('--K', default=150, type=int)
parser.add_argument('--atp', default=0.7, type=int) # DDI
parser.add_argument('--atpe', default=0.55, type=int) # DDI for each epoch
parser.add_argument('--train_step', default=100, type=int)
parser.add_argument('--Maxiter', default=300, type=int)
parser.add_argument('--bound', default=0.005, type=int)
parser.add_argument('--a', default=1, type=int) # paramter for Gaussian kernel
parser.add_argument('--lam', default=0.01, type=int) # paramter for reward
args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
script_name = os.path.basename(__file__)


min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name +'./'
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
'''

class Replay_buffer():
    '''
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.drop = nn.Dropout(p=0.01)


    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = self.drop(a)
        a = F.relu(self.fc2(a))
        a = self.drop(a)
        a = torch.tanh(self.fc3(a)) * self.max_action

        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        self.drop = nn.Dropout(p=0.01)


    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = self.drop(q)
        q = F.relu(self.fc2(q))
        q = self.drop(q)
        q = self.fc3(q)

        return q


class TD3():
    def __init__(self, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration):

        if self.num_training % 100 == 0 & self.num_training != 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done)*(args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    X = np.loadtxt("Circles_X.txt")
    y = np.loadtxt("Circles_y.txt")
    n = X.shape[0]
    d = X.shape[1]
    K = args.K
    train_iter = 0
    train_step = args. train_step
    Maxiter = args.Maxiter
    bound = args.bound
    atpe = args.atpe
    atp = args.atp
    a = args.a
    lam = args.lam



    agent = TD3(d+1, d, bound)
    sn = train_iter
    dist = CDist(X, X)
    graphW, index, sig2_0 = Buildgraph(dist, K, a)
    Ro = graphW.sum(axis=1)
    At = np.zeros(n)
    for i in range(n):
        temp = copy.copy(X[index[i, :], :])
        At[i] = (np.sum((np.abs(np.mean(temp, axis=0) - X[i, :])) ** 2)) ** 0.5

    APt = At / Ro
    min_APt = np.sort(APt)[math.ceil(n*atpe)]
    env = DataEnv(X, y, K, min_APt)
    
    print("====================================")
    print("Collection Experience...")
    print("====================================")
    if args.load: agent.load()
    for it in range(Maxiter+1):
        XO = copy.copy(env.X)
        dist = CDist(env.X, env.X)
        graphW, index, sig2 = Buildgraph(dist, K + 1, a)
        Ro = np.transpose([graphW.sum(axis=1)])
        ep_reward = 0.0
        env.X = maxmin_norm(env.X)
        at = np.zeros(n)
        pt = np.zeros(n)

        for i in range(n):
            temp = copy.copy(XO[index[i, :], :])
            at[i] = (np.sum((np.abs(np.mean(temp, axis=0) - X[i, :])) ** 2)) ** 0.5
            pt[i] = np.sum(graphW[i, index[i, :]])
            apt = at / (pt+1)
        aptt = np.sort(apt)[math.ceil(n*atp)]
        Rol = Ro / np.max(Ro)
        for i in range(n):

            temp = copy.copy(XO[index[i, :], :])
            Ker = copy.copy(graphW[i, index[i, :]])
            ND = copy.copy(temp - np.tile(X[i, :], (K, 1)))
            state1 = np.dot(Ker, ND)
            state = np.hstack((state1, Rol[i]))

            if len(agent.memory.storage) < args.capacity-1:
                if len(agent.memory.storage) % 2 == 0:
                    action = -bound + 2*bound*np.random.random(2)
                else:
                    action = -action
            else:
                action = agent.select_action(state)
                if it < sn:
                    action = action + np.random.normal(-bound/3, bound/3, size=d)
                    action = action.clip(-bound, bound)

            x, r, is_terminal, graph_i, info = env.step(action, i, Ro[i], sig2, index[i, :], lam, state1, aptt, apt[i], False)
            X[i, :] = copy.copy(x)

            ep_reward += r[2]

            temp = copy.copy(XO[index[i, :], :])
            Ker = copy.copy(graph_i[index[i, :]])
            ND = copy.copy(temp - np.tile(X[i, :], (K, 1)))
            next_state1 = np.dot(Ker, ND)
            next_state = np.hstack((next_state1, Rol[i]))
            agent.memory.push((state, next_state, action, r[2], np.float(False)))
        if len(agent.memory.storage) >= args.capacity-1:
            agent.update(40)
        print(f"Episode: {it} Reward: {ep_reward[0] :.3f}")

        if it % args.log_interval == 0:
            agent.save()
        if it == train_iter:
            train_iter = train_iter + train_step
            env.save_data(it)
            env.reset()

if __name__ == '__main__':
    main()
