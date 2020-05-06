import numpy as np
import matplotlib.pyplot as plt

# generate number of measurements
x = np.arange(0, 10, 1)

# values computed from "measured"
y = np.log(x)

# add some error samples from standard normal distribution
xe = 0.1 * np.abs(np.random.randn(len(y)))

# draw and show errorbar
# width: width of the bar
# edgecolor: the color of bar edges
plt.bar(x, y, yerr=xe, width=0.4, edgecolor='b', align='center', ecolor='r',
        color='cyan', label='experiment #1')

# give some explanations
plt.xlabel('# measurement')
plt.ylabel('Measured Values')
plt.title('Measurements')
plt.legend(loc='upper left')

plt.show()
