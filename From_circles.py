from sklearn.datasets import make_circles
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np

# generate 2d classification dataset
X, Y = make_circles(n_samples=4000, noise=0.18, random_state=256, factor=0.3)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))
colors = {0: 'blue', 1: 'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
   group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

np.savetxt("Circles_X.txt", X)
np.savetxt("Circles_y.txt", Y)
#np.loadtxt("Twomoons_X.txt")