from matplotlib.pyplot import *
import numpy as np

# generate different normal distribution
x1 = np.random.normal(30, 3, 100)
x2 = np.random.normal(20, 2, 100)
x3 = np.random.normal(10, 3, 100)

# plot them
plot(x1, label='1st plot')
plot(x2, label='2nd plot')
plot(x3, label='3rd plot')

# generate a legend box
# set a bounding box, start from position(0., 1.02)
# width = 1 and height = .102
# mode: none or expand to allow the legend box to expand
# horizontally filling the axis area
# borderaxespad defines the padding between the axes and the legend border
# loc parameter ranges from 0 to 10, 3 means lower left
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
       mode="expand", borderaxespad=0.)

# annotate an important value
# coordinate system is specified to be the same as the data one
# xytext defines the starting position
annotate("Important value", (55, 20), xycoords='data',
         xytext=(5, 37), arrowprops=dict(arrowstyle='->'))

show()
