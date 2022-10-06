import numpy as np
import fitsne
from matplotlib import pyplot
from pandas import DataFrame
from scipy.io import loadmat
from norm import maxmin_norm
from sklearn.decomposition import PCA

data = loadmat('Mnist_5000')
X = np.double(data['X'])
Y = np.float32(data['Y'])
X = maxmin_norm(X)
estimator = PCA(n_components=20)
X = estimator.fit_transform(X)
X = X.copy(order='C')
#iris = load_iris()
#X = iris.data
X = fitsne.FItSNE(X)
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=(Y[:, 0]-1)))
#df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))
colors = {0: 'red', 1: 'blue', 2: 'green',  3: 'Yellow', 4: 'cyan',
        5: 'darkred', 6: 'darkblue', 7: 'darkgreen',  8: 'orange', 9: 'darkcyan'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    #pyplot.show()
figname = str('iris')
fig.savefig(figname + '.png')