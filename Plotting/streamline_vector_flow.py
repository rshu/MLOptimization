import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

#  get a dense multi-dimensional 'meshgrid'
# the third is step length, which is number of points
# to include between start and stop
# the complex numbers will include the stop point
Y, X = np.mgrid[0:5:100j, 0:5:100j]
U = np.sin(X)
V = np.sin(Y)
print("X")
pprint(X)

print("Y")
pprint(Y)

plt.streamplot(X, Y, U, V)
plt.show()
