import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pdb


def de(func, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)

    # pdb.set_trace()
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    # convert from float to integer
    pop_denorm_convert = np.int_(pop_denorm).tolist()

    fitness = np.asarray([func(index[0], index[1], index[2], index[3], index[4], index[5]) for index in pop_denorm_convert])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trail_denorm_convert = np.int_(trial_denorm).tolist()
            f = func(trail_denorm_convert[0], trail_denorm_convert[1], trail_denorm_convert[2],
                     trail_denorm_convert[3], trail_denorm_convert[4], trail_denorm_convert[5])

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


features = pd.read_csv(r'C:\Users\terry\PycharmProjects\tutorial\Models\temps.csv')
features = pd.get_dummies(features)
labels = np.array(features['actual'])
features = features.drop('actual', axis=1)
feature_list = list(features.columns)
features = np.array(features)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

# Hyperparameters, default, tuning range, description
# threshold, 0.5, [0.01,1], The value to determine defective or not
# max_feature, None, [0.01, 1], The number of features to consider when looking for the best split
# max_leaf_nodes, None, [1,50], Grow trees with max_leaf_nodes in best-first fashion.
# min_sample_split, 2, [2,20], The minimum number of samples required to split an internal node
# min_samples_leaf, 1, [1, 20], The minimum number of samples required to be at a leaf node.
# n_estimators, 100, [50,150], THe number of trees in the forest.


def RF(n_estimators, min_samples_leaf, min_samples_split, max_leaf_nodes, max_features, max_depth):

    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                               max_leaf_nodes=max_leaf_nodes, max_features=max_features, max_depth=max_depth)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')
    return accuracy


# RF(100, 1, 2, 2, 0.01, 5)

result = list(de(RF, bounds=[(50, 150), (1, 20), (2, 20), (2, 50), (1, 2), (1, 10)]))
print(result[-1])
