# optimize the hyperparameters of a Gradient Boosting Machine using
# the Hyperopt library (with the Tree Parzen Estimator algorithm)

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer
import random

MAX_EVALS = 500
N_FOLDS = 10

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

#  This is an imbalanced class problem
plt.hist(labels, edgecolor='k')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Counts of Labels')
plt.show()

# we will use the common classification metric of
# Receiver Operating Characteristic Area Under the Curve (ROC AUC)
# Randomly guessing on a classification problem will yield an ROC AUC of 0.5
# and a perfect classifier has an ROC AUC of 1.0.


# Model with default hyperparameters
model = lgb.LGBMClassifier()
print(model)

start = timer()
model.fit(features, labels)
train_time = timer() - start

predictions = model.predict_proba(test_features)[:, 1]
auc = roc_auc_score(test_labels, predictions)

print('The baseline score on the test set is {:.4f}.'.format(auc))
print('The baseline training time is {:.4f} seconds'.format(train_time))

# Hyperparameter grid
# It's difficult to say ahead of time what choices will work best,
# so we will use a wide range of values centered around the default for most of the hyperparameters.
param_grid = {
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(30, 150)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10))
}

# Subsampling (only applicable with 'goss')
subsample_dist = list(np.linspace(0.5, 1, 100))

plt.hist(param_grid['learning_rate'], color='r', edgecolor='k')
plt.xlabel('Learning Rate', size=14)
plt.ylabel('Count', size=14)
plt.title('Learning Rate Distribution', size=18)
plt.show()

plt.hist(param_grid['num_leaves'], color='m', edgecolor='k')
plt.xlabel('Learning Number of Leaves', size=14)
plt.ylabel('Count', size=14)
plt.title('Number of Leaves Distribution', size=18)
plt.show()

# Randomly sample parameters for gbm
params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
print(params)

# To add a subsample ratio if the boosting_type is not goss
params['subsample'] = random.sample(subsample_dist, 1)[0] if params['boosting_type'] != 'goss' else 1.0
print(params)

# Create a lgb dataset
train_set = lgb.Dataset(features, label=labels)

# Perform cross validation with 10 folds
# we will not actually train this many estimators because we are using early stopping
# to stop training when the validation score has not improved for 100 estimators.
r = lgb.cv(params, train_set, num_boost_round=10000, nfold=10, metrics='auc',
           early_stopping_rounds=100, verbose_eval=False, seed=50)

# Highest score
r_best = np.max(r['auc-mean'])

# Standard deviation of best score
r_best_std = r['auc-stdv'][np.argmax(r['auc-mean'])]

print('The maximium ROC AUC on the validation set was {:.5f} with std of {:.5f}.'.format(r_best, r_best_std))
print('The ideal number of iterations was {}.'.format(np.argmax(r['auc-mean']) + 1))

# Dataframe to hold cv results
random_results = pd.DataFrame(columns=['loss', 'params', 'iteration', 'estimators', 'time'],
                              index=list(range(MAX_EVALS)))


def random_objective(params, iteration, n_folds=N_FOLDS):
    """Random search objective function. Takes in hyperparameters
       and returns a list of results to be saved."""

    start = timer()

    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round=10000, nfold=n_folds,
                        early_stopping_rounds=100, metrics='auc', seed=50)
    end = timer()
    best_score = np.max(cv_results['auc-mean'])

    # Loss must be minimized
    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Return list of results
    return [loss, params, iteration, n_estimators, end - start]


random.seed(50)

# Iterate through the specified number of evaluations
for i in range(MAX_EVALS):

    # Randomly sample parameters for gbm
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}

    print(params)

    if params['boosting_type'] == 'goss':
        # Cannot subsample with goss
        params['subsample'] = 1.0
    else:
        # Subsample supported for gdbt and dart
        params['subsample'] = random.sample(subsample_dist, 1)[0]

    results_list = random_objective(params, i)

    # Add results to next row in dataframe
    random_results.loc[i, :] = results_list

# Sort results by best validation score
random_results.sort_values('loss', ascending=True, inplace=True)
random_results.reset_index(inplace=True, drop=True)
print(random_results.head())

# the highest score and its hyperparameter
print(random_results.loc[0, 'params'])

# Find the best parameters and number of estimators
best_random_params = random_results.loc[0, 'params'].copy()
best_random_estimators = int(random_results.loc[0, 'estimators'])
best_random_model = lgb.LGBMClassifier(n_estimators=best_random_estimators, n_jobs=-1,
                                       objective='binary', **best_random_params, random_state=50)

# Fit on the training data
best_random_model.fit(features, labels)

# Make test predictions
predictions = best_random_model.predict_proba(test_features)[:, 1]

print(
    'The best model from random search scores {:.4f} on the test data.'.format(roc_auc_score(test_labels, predictions)))
print('This was achieved using {} search iterations.'.format(random_results.loc[0, 'iteration']))
