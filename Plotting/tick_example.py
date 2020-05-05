from pylab import *

# get curent axis
ax = gca()

# set view to tight, and maximum number of tick interval to 10
# ax.locator_params(tight=True, nbins=10)

# set the major locator to be a multiple of 10
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))

# generate 100 normal distribution values
ax.plot(np.random.normal(10, .1, 100))

show()
