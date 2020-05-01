import numpy
from pylab import *
from matplotlib.pyplot import *

# possible window type
WINDOWS = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


# if you want ot see just two windows, comment previous line
# and uncomment the following one
# WINDOWS = ['flat', 'hanning']

def smooth(x, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.
    :param x: input signal
    :param window_len: length of smoothing window
    :param window: type of windows
    :return: smoothed signal
    """
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in WINDOWS:
        raise (ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # adding reflected window in front and at the end
    s = numpy.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    # pick windows type and do averaging
    if window == 'flat':
        w = numpy.ones(window_len, 'd')
    else:
        # call approriate function in numpy
        w = eval('numpy.' + window + '(window_len)')

    # NOTE: length(output) != length(input), to correct this
    # return y[(window_len/2-1):-(window_len/2)] instead of just y
    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


# Get some evenly spaced numbers over a specified interval.
t = linspace(-4, 4, 100)

# make some noisy sinusoidal
x = sin(t)
xn = x + randn(len(t)) * 0.1

# smooth it
y = smooth(x)

# window size
ws = 31

subplot(211)
plot(ones(ws))

# plot for every window
for w in WINDOWS[1:]:
    eval('plot(' + w + '(ws))')

# configure axis properties
axis([0, 30, 0, 1.1])

# add legend for every window
legend(WINDOWS)

title("Smoothing windows")

# add second plot
subplot(212)
# draw original signal
plot(x)

# and signal with added noise
plot(xn)

# smooth signal with noise for every possible windowing algorithm
for w in WINDOWS:
    plot(smooth(xn, 10, w))

# add legend for every graph
l = ['original signal', 'signal with noise']
l.extend(WINDOWS)

legend(l)

title("Smoothed signal")
show()
