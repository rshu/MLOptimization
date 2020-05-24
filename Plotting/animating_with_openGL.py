import numpy as np
from mayavi.mlab import *

# produce some nice data.
n_mer, n_long = 6, 11
pi = np.pi
dphi = pi / 1000.0
phi = np.arange(0.0, 2 * pi + 0.5 * dphi, dphi, 'd')
mu = phi * n_mer

x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
z = np.sin(n_long * mu / n_mer) * 0.5

l = plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')
ms = l.mlab_source

for i in range(100):
    x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer + np.pi * (i + 1) / 5.) * 0.5)
    scalars = np.sin(mu + np.pi * (i + 1) / 5)
    ms.set(x=x, scalars=scalars)
