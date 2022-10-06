import gym
from gym import spaces
import numpy as np
from matplotlib import pyplot
from pandas import DataFrame
from Similarity import cos_sim
import copy
from Similarity import CDist
from Similarity import Buildgraph_i
from sklearn import manifold
import scipy.io as sio



class DataEnv(gym.Env):

    def __init__(self, X, y, n_neighbor, max_APt):
        self.OX = copy.copy(X)
        self.max_APt = max_APt
        self.X = X
        self.y = y
        self.n_neighbor = n_neighbor
        self.max_d = np.max(self.X)
        self.min_d = np.min(self.X)
        self.max_step = (self.max_d - self.min_d) * 0.05

        self.action_space = spaces.Box(low=-self.max_step, high=self.max_step,
                                       shape=(1, self.X.shape[1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_d, high=self.max_d,
                                            shape=(self.n_neighbor, self.X.shape[1]), dtype=np.float32)

    def step(self, action, i, ro_i0, sig2, index_i, lam, z, att, at, is_terminal):
        if at < self.max_APt or at < att:
            action_0 = np.zeros(action.shape)
            self.X[i, :] = self.X[i, :] + action_0
        else:
            self.X[i, :] = self.X[i, :] + action

        dist = CDist(np.array([self.X[i, :]]), self.X)
        graph_i = Buildgraph_i(dist, index_i, sig2)
        ro_i = np.sum(graph_i)
        r1 = lam * (ro_i - ro_i0)
        cos = cos_sim(z, action)
        cos = np.nan_to_num(cos)
        r2 = np.sign(cos)*(cos**2)
        r = r1 + r2


        return self.X[i, :], [r1, r2, r], is_terminal, graph_i, {}

    def reset(self):
        self.X = copy.copy(self.OX)

    def get_obs(self, index_i):
        return self.X[[index_i[0, :]], :]

    def save_data(self, figname):
        if self.X.shape[1] <= 2:
            X = self.X
            y = self.y
        else:
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            X = tsne.fit_transform(self.X)
            y = self.y
        if np.min(y) == 1:
            y = y-1
        df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        colors = {0: 'red', 1: 'blue', 2: 'green',  3: 'Yellow', 4: 'cyan',
                  5: 'darkred', 6: 'darkblue', 7: 'darkgreen',  8: 'orange', 9: 'darkcyan'}
        fig, ax = pyplot.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        figname = str(figname)
        fig.savefig(figname + '.png')
        sio.savemat(figname+'.npy', {'X': self.X, 'Y': self.y})