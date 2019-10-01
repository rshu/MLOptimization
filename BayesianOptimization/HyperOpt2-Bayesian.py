import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
import numpy as np
import pandas as pd

# Modeling
import lightgbm as lgb

from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import matplotlib.pyplot as plt
import seaborn as sns

from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

import ast
from sklearn.metrics import roc_auc_score

# For Bayesian optimization, we need the following four parts:
#
# Objective function
# Domain space
# Hyperparameter optimization algorithm
# History of results

N_FOLDS = 10
MAX_EVALS = 500
out_file = './gbm_trials.csv'

# Read in data and separate into training and testing sets
data = pd.read_csv('./data/caravan-insurance-challenge.csv')
train = data[data['ORIGIN'] == 'train']
test = data[data['ORIGIN'] == 'test']

# Extract the labels and format properly
train_labels = np.array(train['CARAVAN'].astype(np.int32)).reshape((-1,))
test_labels = np.array(test['CARAVAN'].astype(np.int32)).reshape((-1,))

# Drop the unneeded columns
train = train.drop(columns=['ORIGIN', 'CARAVAN'])
test = test.drop(columns=['ORIGIN', 'CARAVAN'])

# Convert to numpy array for splitting in cross validation
features = np.array(train)
test_features = np.array(test)
labels = train_labels[:]

print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
print(train.head())

# Create a lgb dataset
train_set = lgb.Dataset(features, label=labels)


def objective(params, n_folds=N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])

    start = timer()

    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round=10000, nfold=n_folds,
                        early_stopping_rounds=100, metrics='auc', seed=50)

    run_time = timer() - start

    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])

    # Hyperopt works to minimize a function
    # Loss must be minimized
    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'train_time': run_time, 'status': STATUS_OK}


# Create the learning rate
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}

learning_rate_dist = []

# Draw 10000 samples from the learning rate domain
for _ in range(10000):
    learning_rate_dist.append(sample(learning_rate)['learning_rate'])

plt.figure(figsize=(8, 6))
sns.kdeplot(learning_rate_dist, color='red', linewidth=2, shade=True)
plt.title('Learning Rate Distribution', size=18)
plt.xlabel('Learning Rate', size=16)
plt.ylabel('Density', size=16)
plt.show()

# Discrete uniform distribution
num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
num_leaves_dist = []

# Sample 10000 times from the number of leaves distribution
for _ in range(10000):
    num_leaves_dist.append(sample(num_leaves)['num_leaves'])

# kdeplot
plt.figure(figsize=(8, 6))
sns.kdeplot(num_leaves_dist, linewidth=2, shade=True)
plt.title('Number of Leaves Distribution', size=18)
plt.xlabel('Number of Leaves', size=16)
plt.ylabel('Density', size=16)
plt.show()

# # boosting type domain
# # nested conditional statements to indicate hyperparameters that depend on other hyperparameters
# # we can explore different models with completely different sets of hyperparameters by using nested conditionals.
# boosting_type = {'boosting_type': hp.choice('boosting_type',
#                                             [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)},
#                                              {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
#                                              {'boosting_type': 'goss', 'subsample': 1.0}])}
#
# # Draw a sample
# params = sample(boosting_type)
# print(params)

# # Retrieve the subsample if present otherwise set to 1.0
# subsample = params['boosting_type'].get('subsample', 1.0)
#
# # Extract the boosting type
# params['boosting_type'] = params['boosting_type']['boosting_type']
# params['subsample'] = subsample
# print(params)


# Define the entire search space
# In Hyperopt, and other Bayesian optimization frameworks,
# the domain is not a discrete grid but instead has probability
# distributions for each hyperparameter.
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

# Sample from the full space
x = sample(space)

# Conditional logic to assign top-level keys
# Every time we run this code, the results will change.
subsample = x['boosting_type'].get('subsample', 1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample
print(x)

# optimization algorithm
# Tree Parzen Estimator
# the method for constructing the surrogate function and
# choosing the next hyperparameters to evaluate.
tpe_algorithm = tpe.suggest

# Keep track of results
bayes_trials = Trials()

# File to save first results
# Every time the objective function is called, it will write one line to this file.
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Global variable
global ITERATION

ITERATION = 0

# Run optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(50))

# Sort the trials with lowest loss (highest AUC) first
bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
print(bayes_trials_results[:2])

results = pd.read_csv('./gbm_trials.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)
print(results.head())

# Convert from a string to a dictionary
ast.literal_eval(results.loc[0, 'params'])

# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, n_jobs=-1,
                                      objective='binary', random_state=50, **best_bayes_params)
best_bayes_model.fit(features, labels)

# Evaluate on the testing data
preds = best_bayes_model.predict_proba(test_features)[:, 1]
print('The best model from Bayes optimization scores {:.5f} AUC ROC on the test set.'.format(
    roc_auc_score(test_labels, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))