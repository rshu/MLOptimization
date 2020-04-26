from pylab import *

# fake up some data
spread = rand(50) * 100
center = ones(25) * 50

# generate some outliers high and low
filter_high = rand(10) * 100 + 100
filter_low = rand(10) * -100

# merge generated data set
data = concatenate((spread, center, filter_high, filter_low), 0)

subplot(311)

# basic plot
# 'gx' defines the outlier plotting properties
boxplot(data, 0, 'gx')

# compare this with similar scatter plots
subplot(312)
# join arrays, axis = 0
spread_1 = concatenate((spread, filter_high, filter_low), 0)
center_1 = ones(70) * 25
# A scatter plot of y vs. x with varying marker size and/or color. (x, y)
scatter(center_1, spread_1)
xlim([0, 50])

# and with another that is more approriate for
# scatter plot
subplot(313)
center_2 = rand(70) * 50
scatter(center_2, spread_1)
xlim([0, 50])

show()
