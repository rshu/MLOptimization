import pylab
import random

SAMPLE_SIZE = 100

# seed random generator
# if not argument provided
# uses system current time
random.seed()

# store generated random values hers
real_rand_vars = []

# pick some random values
read_rand_vars = [random.randint(1,100) for val in range(SAMPLE_SIZE)]

# create histogram from data in 10 buckets
pylab.hist(read_rand_vars, 10)

# define x and y labels
pylab.xlabel("Number Range")
pylab.ylabel("Count")

# show figure
pylab.show()
