import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def de(func, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)

    # popsize individuals with dimension params each
    pop = np.random.rand(popsize, dimensions)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)

    # convert each component from [0,1] to [min,max]
    pop_denorm = min_b + pop * diff

    # save evaluation of the initial population
    fitness = np.asarray([func(ind) for ind in pop_denorm])

    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):

            # select three other vectors that are not the current one
            idxs = [idx for idx in range(popsize) if idx != j]

            # randomly choose 3 indexes without replacement
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]

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

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff

            f = func(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield min_b + pop * diff, fitness, best_idx

x = np.linspace(0, 10, 500)
y = np.cos(x) + np.random.normal(0, 0.2, 500)
plt.scatter(x, y)
plt.plot(x, np.cos(x), label='cos(x)', color='red')
plt.legend()
plt.show()


def fmodel(x, w):
    return w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 + w[5] * x**5


plt.plot(x, fmodel(x, [1.0, -0.01, 0.01, -0.1, 0.1, -0.01]))
plt.show()

# Root Mean Square Error (RMSE)
def rmse(w):
    y_pred = fmodel(x, w)
    return np.sqrt(sum((y - y_pred)**2) / len(y))


result = list(de(rmse, [(-5, 5)] * 6, its=2000))
# fig, ax = plt.subplots()
print(result)

plt.scatter(x, y)
plt.plot(x, np.cos(x), label='cos(x)', color='red')
plt.plot(x, fmodel(x, [0.99677643, 0.47572443, -1.39088333,
                       0.50950016, -0.06498931, 0.00273167]), label='result', color='green')
plt.legend()
plt.show()


# def animate(i):
#     ax.clear()
#     ax.set_ylim([-2, 2])
#     ax.scatter(x, y)
#     pop, fit, idx = result[i]
#     for ind in pop:
#         data = fmodel(x, ind)
#         ax.plot(x, data, alpha=0.3)
#
# anim = animation.FuncAnimation(fig, animate, frames=2000, interval=20)
# HTML(anim.to_html5_video())
