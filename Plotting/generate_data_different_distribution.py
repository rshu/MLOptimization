import random
import matplotlib
import matplotlib.pyplot as plt

SAMPLE_SIZE = 1000
# histogram buckets
buckets = 100

plt.figure()

# we need to update font size just for this example
matplotlib.rcParams.update({'font.size': 7})

# define a grid of six by two subplots
plt.subplot(621)
plt.xlabel("random.random")
# Return the next random floating point number in the range [0.0, 1.0]
res = [random.random() for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.subplot(622)
plt.xlabel("random.uniform")
# Return a random floating point number N such that a <= N <= b
# for a <= b and b <= N <= a for b < a
# The end-point value b may or may not be included in the range
# depending on floating-point rounding in the equation a + (b-a)*random()
a = 1
b = SAMPLE_SIZE
res = [random.uniform(a, b) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.subplot(623)
plt.xlabel("random.triangular")
# Return a random floating point number N such that low <= N <= high
# and with the specified mode between those bounds
# The low and high bounds default to zero and one
# The mode argument defaults to the midpoint between the bounds, giving
# a symmetric distribution
low = 1
high = SAMPLE_SIZE
res = [random.triangular(low, high) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.subplot(624)
plt.xlabel("random.betavariate")
alpha = 1
beta = 10
res = [random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.subplot(625)
plt.xlabel("random.expovaraite")
lambd = 1.0 / ((SAMPLE_SIZE + 1) / 2.)
res = [random.expovariate(lambd) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.subplot(626)
plt.xlabel("random.gamavariate")
alpha = 1
beta = 10
res = [random.gammavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.subplot(627)
plt.xlabel("random.lognormvariate")
mu = 1
sigma = 0.5
res = [random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.subplot(628)
plt.xlabel("random.normalvariate")
mu = 1
sigma = 0.5
res = [random.normalvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.subplot(629)
plt.xlabel("random.paretovariate")
alpha = 1
res = [random.paretovariate(alpha) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.tight_layout()
plt.show()
