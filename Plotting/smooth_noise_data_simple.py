from pylab import *
import numpy as np


def moving_average(interval, window_size):
    '''
    Compute convoluted window for given size
    '''
    window = np.ones(int(window_size)) / float(window_size)
    # return a discrete linear convolution of two one-dimensional sequences
    return np.convolve(interval, window, 'same')


# generate a sequence of evenly spaced numbers for a specified interval
t = np.linspace(-4, 4, 100)
y = np.sin(t) + randn(len(t)) * 0.1

plot(t, y, "k.")

# compute moving average
y_av = moving_average(y, 10)
plot(t, y_av, "r")

# xlim(0, 1000)
xlabel("Time")
ylabel("Value")
grid(True)
show()
