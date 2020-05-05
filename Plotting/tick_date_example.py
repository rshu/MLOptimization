from pylab import *
import matplotlib as mpl
import datetime
import numpy as np

# explicitly create a figure
fig = figure()

# get current axis
ax = gca()

# set some date range
start = datetime.datetime(2013, 1, 1)
stop = datetime.datetime(2013, 12, 1)
delta = datetime.timedelta(days=1)

# convert date for matplotlib
dates = mpl.dates.drange(start, stop, delta)

# generate some random values
values = np.random.rand(len(dates))

ax = gca()

# create plot with dates
ax.plot_date(dates, values, linestyle='-', marker='')

# specify formatter
date_format = mpl.dates.DateFormatter('%Y-%m-%d')

# apply formatter
ax.xaxis.set_major_formatter(date_format)

# autoformat date labels
# rotates labels by 30 degrees by default
# use rotate param to specify different rotation degree
# use bottom param to give some room to data labels
fig.autofmt_xdate()

show()
