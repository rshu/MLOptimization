from pylab import *
import matplotlib.pyplot as plt

# make a square figure and axes
figure(1, figsize=(6, 6))
# ax = plt.axes([0.1, 0.1, 0.8, 0.8], facecolor='k')

# the slices will be ordered
# and plotted counter-clockwise
labels = 'Spring', 'Summer', 'Autumn', 'Winter'

# fractions are either x/sum(x) or x if sum(x) <= 1
x = [20.0, 21.3, 21.3, 37.3]

# explode must be len(x) sequence or None
# offset to the arc
explode = (0.1, 0.1, 0.3, 0.1)

# autopct to format the label
# startangle: counterclockwise from x axis, angle 0
pie(x, explode=explode, labels=labels,
    autopct='%1.1f%%', startangle=67, shadow=True)

title('Rainy days by season')
show()
