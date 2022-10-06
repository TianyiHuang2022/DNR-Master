import numpy as np


def maxmin_norm(Data):
    n_dim = Data.shape[1]
    max_D = np.max(Data, axis=0)
    min_D = np.min(Data, axis=0)
    temp = max_D - min_D
    for i in range(n_dim):
        if temp[i] != 0:
            Data[:, i] = (Data[:, i] - min_D[i]) / temp[i]
    return Data