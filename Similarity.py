import numpy as np
import numpy as np
from scipy import *

def CDist(data1, data2):
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1_s = np.sum(data1**2, axis=1)
    data2_s = np.sum(data2**2, axis=1)
    dist = np.tile(np.transpose([data1_s]), (1, n2)) + np.tile([data2_s], (n1, 1)) - 2 * np.dot(data1,
                                                                                                np.transpose(data2))
    Dist = dist**0.5
    Dist = np.nan_to_num(Dist)
    Dist = np.maximum(Dist, 0)
    Dist = np.around(Dist, decimals=4)
    return Dist


def CKROD(Dist, sigma):
    Dist = Dist / np.max(Dist)
    Rank = Dist.argsort().argsort()
    Rdist = Rank + np.transpose(Rank)
    Rdist = Rdist / np.max(Rdist)
    KROD = Rdist * np.exp(Dist**2 / sigma)
    return KROD


def Buildgraph(distance_matrix, K, a):
    #distance_matrix = distance_matrix**2
    n_sample = distance_matrix.shape[0]
    index = distance_matrix.argsort()
    index = index[:, 1:K]
    sortedDist = np.sort(distance_matrix)
    sig2 = np.mean(sortedDist[:, 1:K]) * a
    ND = sortedDist[:, 1:K]
    ND = np.exp(- ND / sig2)
    graph = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(K-1):
            graph[i, index[i, j]] = ND[i, j]
    return graph, index, sig2


def Buildgraph_i(distance_matrix_i, index_i, sig2):
    #distance_matrix_i = distance_matrix_i**2
    n_sample = distance_matrix_i.shape[1]
    K = index_i.shape[0]
    ND = distance_matrix_i[0, index_i[:]]
    ND = np.exp(- ND / sig2)
    graph_i = np.zeros(n_sample)
    for j in range(K):
        graph_i[index_i[j]] = ND[j]
    return graph_i


def cos_sim(vector_a, vector_b):

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim

def per_sim(vector_a, vector_b):

    vector_a = np.mat(vector_a)-np.mat(np.mean(vector_a))
    vector_b = np.mat(vector_b)-np.mat(np.mean(vector_b))
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        sim = 0
    else:
        sim = num / (denom)
    return sim

def asymmetricKL(P,Q):
    return sum(P * log(P / Q)) #calculate the kl divergence between P and Q

def symmetricalKL(P,Q):
    return (asymmetricKL(P,Q)+asymmetricKL(Q,P))/2.00