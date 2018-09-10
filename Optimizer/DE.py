import numpy as np
from yabox.problems import Levy
import matplotlib.pyplot as plt


# function to optimize
def fobj(x):
    value = 0
    for i in range(len(x)):
        value += x[i]**2
    return value / len(x)


def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


it = list(de(lambda x: x**2, bounds=[(-100, 100)]))
print(it[-1])
# (array([0.]), array([0.]))
#  just a single number since the function is 1-D


# problem = Levy()
# problem.plot3d()

# result = list(de(lambda x: x**2 / len(x), bounds=[(-100, 100)] * 32, its=3000))
# print(result[-1])
# x, f = zip(*result)
# plt.plot(f)

for d in [8, 16, 32, 64]:
    it = list(de(lambda x: sum(x**2)/d, [(-100, 100)] * d, its=3000))
    x, f = zip(*it)
    plt.plot(f, label='d={}'.format(d))
plt.xlabel('Iterations')
plt.ylabel('f(x)')
plt.legend()
plt.show()
