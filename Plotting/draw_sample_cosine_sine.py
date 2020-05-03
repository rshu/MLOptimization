import matplotlib.pyplot as plt
import numpy as np

# generate uniformly distributed
# 256 points from -pi to pi, inclusive
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)

# compute sine and cos for every x
y = np.cos(x)
y1 = np.sin(x)

plt.plot(x, y)
plt.plot(x, y1)

plt.title("Functions $\sin$ and $\cos$")

# set x limit
plt.xlim(-3.0, 3.0)

# set y limit
plt.ylim(-1.0, 1.0)

# format ticks at specific values
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1, 0, +1], [r'$-1$', r'$0$', r'$+1$'])

plt.show()
