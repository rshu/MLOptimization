import numpy as np
from yabox.problems import Levy
import matplotlib.pyplot as plt


def de(func, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)

    # popsize individuals with dimension params each
    pop = np.random.rand(popsize, dimensions)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)

    # convert each component from [0,1] to [min,max]
    pop_denorm = min_b + pop * diff

    # save evaluation of the initial population
    fitness = np.asarray([func(index) for index in pop_denorm])

    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):

            # select three other vectors that are not the current one
            idxs = [idx for idx in range(popsize) if idx != j]

            # randomly choose 3 indexes without replacement (unique sample)
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            # A larger mutation factor increases the search radius but may
            # slowdown the convergence of the algorithm.
            # Values for mut are usually chosen from the interval [0.5, 2.0]
            # clipping the number to the interval
            # values greater than 1 become 1, and the values smaller than 0 become 0
            mutant = np.clip(a + mut * (b - c), 0, 1)

            # Recombination is about mixing the information of the mutant
            # with the information of the current vector to create a trial vector.
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            # take values from mutant vector when true
            trial = np.where(cross_points, mutant, pop[j])

            # denormalize the trail vector to [min, max]
            trial_denorm = min_b + trial * diff

            f = func(trial_denorm)

            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


# it = list(de(lambda x: x**2, bounds=[(-100, 100)]))
# print(it[-1])
# (array([0.]), array([0.]))
#  just a single number since the function is 1-D

# function to optimize
# def func(x):
#     value = 0
#     for i in range(len(x)):
#         value += x[i]**2
#     return value / len(x)

# problem = Levy()
# problem.plot3d()

# 2 dimensional function
# result = list(de(lambda x: x**2 / len(x), bounds=[(-100, 100)] * 32))
# print(result[-1])

# using 3000 iterations instead of default 1000 iterations
# result = list(de(lambda x: x**2 / len(x), bounds=[(-100, 100)]*32, its=3000))
# print(result[-1])

# (array([0., 0., 0., 0., 0., 0., 0., 0.]), 0.0)
result = list(de(lambda x: sum(x**2)/len(x), [(-100, 100)] * 8, its=3000))
print(result[-1])
x, f = zip(*result)
plt.plot(f)
plt.show()


# when the number of dimensions grows, the number of iterations required
# by the algorithm to find a good solution grows as well
for d in [8, 16, 32, 64]:
    it = list(de(lambda x: sum(x**2)/d, [(-100, 100)] * d, its=3000))
    x, f = zip(*it)
    plt.plot(f, label='d={}'.format(d))
plt.xlabel('Iterations')
plt.ylabel('f(x)')
plt.legend()
plt.show()


