import matplotlib.pyplot as plt
import numpy as np
from add_histogram_to_scatter_plot import DATA as d

total = sum(d)
av = total / len(d)
z = [i - av for i in d]

# Now let's generate random data for the same period
d1 = np.random.random(365)
assert len(d) == len(d1)

total1 = sum(d1)
av1 = total1 / len(d1)
z1 = [i - av1 for i in d1]

fig = plt.figure()

# Search trend volume
ax1 = fig.add_subplot(311)
ax1.plot(d)
ax1.set_xlabel('Google Trends data for "flowers"')

ax2 = fig.add_subplot(312)
ax2.plot(d1)
ax2.set_xlabel('Random Data')

ax3 = fig.add_subplot(313)
ax3.set_xlabel('Cross correlation of random data')
ax3.xcorr(z, z1, usevlines=True, maxlags=None, normed=True, lw=2)
ax3.grid(True)

plt.ylim(-1, 1)
plt.tight_layout()
plt.show()

# the cross-correlation diagram tells that these two datasets are not correlated
# which is represented by the height of correlation values
