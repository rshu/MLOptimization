import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pdb
import matplotlib.pyplot as plt

# 2 dimensional function
# result = list(de(lambda x: x**2 / len(x), bounds=[(-100, 100)]))
# print(result[-1])


def de(func, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    print("AAA")

    # popsize individuals with dimension params each
    pop = np.random.rand(popsize, dimensions)
    pdb.set_trace()

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)

    # convert each component from [0,1] to [min,max]
    pop_denorm = min_b + pop * diff
    #...

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


""" In the case of a random forest, hyperparameters
include the number of decision trees in the forest
and the number of features considered by each tree
when splitting a node. """

"""The parameters of a random forest are the 
variables and thresholds used to split each node
learned during training"""

# Read in data and display first 5 rows
features = pd.read_csv(r'C:\Users\terry\PycharmProjects\tutorial\Models\temps.csv')

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('actual', axis=1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

# Hyperparameters, default, tuning range, description
# threshold, 0.5, [0.01,1], The value to determine defective or not
# max_feature, None, [0.01, 1], The number of features to consider when looking for the best split
# max_leaf_nodes, None, [1,50], Grow trees with max_leaf_nodes in best-first fashion.
# min_sample_split, 2, [2,20], The minimum number of samples required to split an internal node
# min_samples_leaf, 1, [1, 20], The minimum number of samples required to be at a leaf node.
# n_estimators, 100, [50,150], THe number of trees in the forest.

print("BBB")


def RF(n_estimators=100, min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=25, max_features=0.01, max_depth=5):
    rf = RandomForestRegressor(n_estimators, min_samples_leaf, min_samples_split, max_leaf_nodes,
                               max_features, max_depth)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')
    return accuracy

print("CCC")


result = list(de(RF, bounds=[(50, 150), (1,20), (2,20), (1,50), (0.1,1), (1,10)]))
print(result[-1])


# for n_tree in range(500, 1500):
#     for max_depth in range(3, 7):
#         rffunction(n_tree, max_depth)





