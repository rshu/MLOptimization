import numpy as np
from hyperopt import hp, tpe, fmin, rand
from hyperopt import Trials
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK
from timeit import default_timer as timer


# # Single line bayesian optimization of polynomial function
# # A one-dimensional polynomial class.
# # hp.normal(label, mu, sigma)
# # Returns a real value that's normally-distributed with mean mu and standard deviation sigma.
# # When optimizing, this is an unconstrained variable.
# best = fmin(fn=lambda x: np.poly1d([1, -2, -28, 28, 12, -26, 100])(x),
#             space=hp.normal('x', 4.9, 0.5),
#             algo=tpe.suggest,
#             max_evals=2000)
# print(best)


# a simple polynomial function with the goal being to find the minimum value
def objective(x):
    """Objective function to minimize"""

    # Create the polynomial object
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])

    # Return the value of the polynomial
    return f(x) * 0.05


# Space over which to evluate the function is -5 to 6
x = np.linspace(-5, 6, 10000)
y = objective(x)

miny = min(y)
minx = x[np.argmin(y)]

# Visualize the function
plt.figure(figsize=(8, 6))
plt.style.use('fivethirtyeight')
plt.title('Objective Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.vlines(minx, min(y) - 50, max(y), linestyles='--', colors='r')
plt.plot(x, y)
plt.show()

# Print out the minimum of the function and value
print('Minimum of %0.4f occurs at %0.4f' % (miny, minx))

# Create the domain space
# hp.uniform(label, low, high)
# Returns a value uniformly between low and high.
space = hp.uniform('x', -5, 6)

samples = []

# Sample 10000 values from the range
for _ in range(10000):
    samples.append(sample(space))

# Histogram of the values
plt.hist(samples, bins=20, edgecolor='black')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Domain Space')
plt.show()

# Create the algorithm
tpe_algo = tpe.suggest
rand_algo = rand.suggest

# Create a trials object
# Hyperopt keeps track of the results for the algorithm internally.
# if we want to inspect the progression of the alogorithm,
# we need to create a Trials object that will record the values and the scores
tpe_trials = Trials()
rand_trials = Trials()

# Run 2000 evals with the tpe algorithm
# Here we ran the same number of totals calls, but this was probably not necessary
# because the Tree Parzen Estimator will converge on the optimum quickly without
# the need for more iterations.
tpe_best = fmin(fn=objective, space=space,
                algo=tpe_algo, trials=tpe_trials,
                max_evals=2000)

# minimize the loss function
print(tpe_best)

# Run 2000 evals with the random algorithm
rand_best = fmin(fn=objective, space=space, algo=rand_algo, trials=rand_trials,
                 max_evals=2000, rstate=np.random.RandomState(50))

# Print out information about losses
print('Minimum loss attained with TPE:    {:.4f}'.format(tpe_trials.best_trial['result']['loss']))
print('Minimum loss attained with random: {:.4f}'.format(rand_trials.best_trial['result']['loss']))
print('Actual minimum of f(x):            {:.4f}'.format(miny))

# Print out information about number of trials
print(
    '\nNumber of trials needed to attain minimum with TPE:    {}'.format(tpe_trials.best_trial['misc']['idxs']['x'][0]))
print(
    'Number of trials needed to attain minimum with random: {}'.format(rand_trials.best_trial['misc']['idxs']['x'][0]))

# Print out information about value of x
print('\nBest value of x from TPE:    {:.4f}'.format(tpe_best['x']))
print('Best value of x from random: {:.4f}'.format(rand_best['x']))
print('Actual best value of x:      {:.4f}'.format(minx))

# Dataframe of results from optimization
tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
                            'iteration': tpe_trials.idxs_vals[0]['x'],
                            'x': tpe_trials.idxs_vals[1]['x']})

print(tpe_results.head())

tpe_results['rolling_average_x'] = tpe_results['x'].rolling(50).mean().fillna(method='bfill')
tpe_results['rolling_average_loss'] = tpe_results['loss'].rolling(50).mean().fillna(method='bfill')
print(tpe_results.head())

plt.figure(figsize=(10, 8))
plt.plot(tpe_results['iteration'], tpe_results['x'], 'bo', alpha=0.5)
plt.xlabel('Iteration', size=22)
plt.ylabel('x value', size=22)
plt.title('TPE Sequence of Values', size=24)
plt.hlines(minx, 0, 2000, linestyles='--', colors='r')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(tpe_results['x'], bins=50, edgecolor='k')
plt.title('Histogram of TPE Values')
plt.xlabel('Value of x')
plt.ylabel('Count')
plt.show()

# Sort with best loss first
tpe_results = tpe_results.sort_values('loss', ascending=True).reset_index()

plt.plot(tpe_results['iteration'], tpe_results['loss'], 'bo', alpha=0.3)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('TPE Sequence of Losses')
plt.show()

print('Best Loss of {:.4f} occured at iteration {}'.format(tpe_results['loss'][0], tpe_results['iteration'][0]))

rand_results = pd.DataFrame(
    {'loss': [x['loss'] for x in rand_trials.results], 'iteration': rand_trials.idxs_vals[0]['x'],
     'x': rand_trials.idxs_vals[1]['x']})

print(rand_results.head())

plt.figure(figsize=(10, 8))
plt.plot(rand_results['iteration'], rand_results['x'], 'bo', alpha=0.5)
plt.xlabel('Iteration', size=22)
plt.ylabel('x value', size=22)
plt.title('Random Sequence of Values', size=24)
plt.hlines(minx, 0, 2000, linestyles='--', colors='r')
plt.show()

# Sort with best loss first
rand_results = rand_results.sort_values('loss', ascending=True).reset_index()

plt.figure(figsize=(8, 6))
plt.hist(rand_results['x'], bins=50, edgecolor='k')
plt.title('Histogram of Random Values')
plt.xlabel('Value of x')
plt.ylabel('Count')
plt.show()

# Print information
print('Best Loss of {:.4f} occured at iteration {}'.format(rand_results['loss'][0], rand_results['iteration'][0]))

# Normally distributed space
space = hp.normal('x', 4.9, 0.5)

samples = []

# Sample 10000 values from the range
for _ in range(10000):
    samples.append(sample(space))

# Histogram of the values
plt.hist(samples, bins=20, edgecolor='black')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Domain Space')
plt.show()


def objective(x):
    """Objective function to minimize with smarter return values"""

    # Create the polynomial object
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])

    # Evaluate the function
    start = timer()
    loss = f(x) * 0.05
    end = timer()

    # Calculate time to evaluate
    time_elapsed = end - start

    results = {'loss': loss, 'status': STATUS_OK, 'x': x, 'time': time_elapsed}

    # Return dictionary
    return results


# New trials object
trials = Trials()

# Run 2000 evals with the tpe algorithm
best = fmin(fn=objective, space=space, algo=tpe_algo, trials=trials,
            max_evals=2000, rstate=np.random.RandomState(120))

results = trials.results
print(results[:2])

# Results into a dataframe
results_df = pd.DataFrame({'time': [x['time'] for x in results],
                           'loss': [x['loss'] for x in results],
                           'x': [x['x'] for x in results],
                           'iteration': list(range(len(results)))})

# Sort with lowest loss on top
results_df = results_df.sort_values('loss', ascending=True)
print(results_df.head())

plt.hist(results_df['x'], bins=50, edgecolor='k')
plt.title('Histogram of TPE Values')
plt.xlabel('Value of x')
plt.ylabel('Count')
plt.show()

sns.kdeplot(results_df['x'], label='Normal Domain')
sns.kdeplot(tpe_results['x'], label='Uniform Domain')
plt.legend()
plt.xlabel('Value of x')
plt.ylabel('Density')
plt.title('Comparison of Domain Choice using TPE')
plt.show()

print('Lowest Value of the Objective Function = {:.4f} at x = {:.4f} found in {:.0f} iterations.'.format(
    results_df['loss'].min(),
    results_df.loc[results_df['loss'].idxmin()]['x'],
    results_df.loc[results_df['loss'].idxmin()]['iteration']))

