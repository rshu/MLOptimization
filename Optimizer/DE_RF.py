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
    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(np.int_(np.round_(index[0])))
        temp_list.append(np.int_(np.round_(index[1])))
        temp_list.append(np.int_(np.round_(index[2])))
        temp_list.append(np.int_(np.round_(index[3])))
        temp_list.append(float('%.2f' % index[4]))
        temp_list.append(np.int(np.round_(index[5])))
        result_list.append(temp_list)
        temp_list = []

    fitness = np.asarray([func(index[0], index[1], index[2], index[3], index[4], index[5])
                          for index in result_list])

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
            trail_denorm_convert = trial_denorm.tolist()
            f = func(np.int_(np.round_(trail_denorm_convert[0])), np.int_(np.round_(trail_denorm_convert[1])), np.int_(np.round_(trail_denorm_convert[2])),
                     np.int_(np.round_(trail_denorm_convert[3])), float('%.2f' % trail_denorm_convert[4]), np.int_(np.round_(trail_denorm_convert[5])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


# Hyperparameters, default, tuning range, description
# threshold, 0.5, [0.01,1], The value to determine defective or not
# max_feature, None, [0.01, 1], The number of features to consider when looking for the best split
# max_leaf_nodes, None, [1,50], Grow trees with max_leaf_nodes in best-first fashion.
# min_sample_split, 2, [2,20], The minimum number of samples required to split an internal node
# min_samples_leaf, 1, [1, 20], The minimum number of samples required to be at a leaf node.
# n_estimators, 100, [50,150], THe number of trees in the forest.

features = pd.read_csv(r'C:\Users\Terry\PycharmProjects\MLOptimization\Models\temps.csv')
features = pd.get_dummies(features)
labels = np.array(features['actual'])
features = features.drop('actual', axis=1)
feature_list = list(features.columns)
features = np.array(features)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)


def rf(n_estimators, min_samples_leaf, min_samples_split, max_leaf_nodes, max_features, max_depth):

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
result = list(de(rf, bounds=[(50, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]))
print(result[-1])

# (array([54.44211646,  1., 10.38390383, 28.67384164,  0.25510296, 9.96982901]), 94.53515416736293)
# (array([50., 6.57443301, 2., 35.97013398, 0.3746776, 9.16419074]), 94.53879111488068)
# (array([50., 4.24348019,  5.20492519, 39.79956237,  0.25259712,10.]), 94.53370431667776)
# (array([78.2370243, 3.65769774, 9.42066649, 23.58407648,  0.31318909, 10.]), 94.55013371811786)
# (array([52.27037848,  5.44928123,  6.19726546, 50.        ,  0.35213438, 6.36864629]), 94.52523717448143)
# (array([50., 7.64186483, 2., 44.37173323, 0.33157424, 5.12368077]), 94.48520255569085)
# (array([50., 7.91581119, 15.51653384, 41.36672505,  0.46084965, 6.27259968]), 94.51749622161212)
# (array([50.,  1., 13.4793573 , 50.,  0.36738165, 5.81136957]), 94.52214877481761)
