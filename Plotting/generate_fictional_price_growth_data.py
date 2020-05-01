import pylab
import random

# days to generate data for
duartion = 100

# mean value
mean_inc = 0.2

# standard deviation
std_dev_inc = 1.2

# time series
x = range(duartion)
y = []

# IPO price
price_today = 45

for i in x:
    next_delta = random.normalvariate(mean_inc, std_dev_inc)
    price_today += next_delta
    y.append(price_today)

pylab.plot(x, y)
pylab.xlabel("Time")
pylab.ylabel("Price")
pylab.show()
