import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

MAX_EVALS = 2
N_FOLDS = 2

# Read in data and separate into training and testing sets
features = pd.read_csv('./data/application_train.csv')

# Extract the labels and format properly
labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1,))

# Drop the unneeded columns
features = features.drop(columns=['SK_ID_CURR', 'TARGET'])

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

print('Train shape: ', train_features.shape)
print('Test shape: ', test_features.shape)

print(train_features.head())

model = lgb.LGBMClassifier(random_state=50)

one_hot_train = pd.get_dummies(train_features)
one_hot_test = pd.get_dummies(test_features)

one_hot_train, one_hot_test = one_hot_train.align(one_hot_test, axis=1, join='inner')
one_hot_features = list(one_hot_train.columns)

print('Number of features: ', one_hot_train.shape[1])

start = timer()
cv = cross_val_score(estimator=model, X=one_hot_train,
                     y=train_labels,
                     verbose=2, n_jobs=-1,
                     cv=N_FOLDS, scoring='roc_auc')
cv_time = timer() - start

print('Baseline model {} fold cv AUC ROC score: {:.5f}'.format(N_FOLDS, np.mean(cv)))
print('Baseline model eval time: {:.2f} seconds.'.format(cv_time))

# label encoding
le = LabelEncoder()

le_train = train_features.copy()
le_test = test_features.copy()

cat_features = []

for i, col in enumerate(le_train):
    if le_train[col].dtype == 'object':
        le_train[col] = le.fit_transform(np.array(le_train[col].astype(str)).reshape((-1,)))
        le_test[col] = le.transform(np.array(le_test[col].astype(str)).reshape((-1,)))
        cat_features.append(i)

print('Number of features: ', le_train.shape[1])

start = timer()
cv = cross_val_score(estimator=model, X=le_train, y=train_labels,
                     fit_params={'categorical_feature': cat_features},
                     verbose=2, n_jobs=-1, cv=N_FOLDS,
                     scoring='roc_auc')
cv_time = timer() - start

print('Baseline model with label encoding {} fold cv AUC ROC score: {:.5f}'.format(N_FOLDS, np.mean(cv)))
print('Baseline model with label encoding eval time: {:.2f} seconds.'.format(cv_time))

# random search
# Hyperparameter grid
param_grid = {
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss'],
    'num_leaves': list(range(30, 150)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'encoding': ['one_hot', 'label']
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

params['subsample'] = random.sample(subsample_dist, 1)[0] if params['boosting_type'] != 'goss' else 1.0
print(params)

# Convert to numpy array for splitting in cross validation
one_hot_features = np.array(one_hot_train)
one_hot_features_test = np.array(one_hot_test)

le_features = np.array(le_train)
le_features_test = np.array(le_test)

labels = train_labels[:]
labels_test = test_labels[:]

# Create a lgb dataset
oh_train_set = lgb.Dataset(one_hot_features, label=train_labels)
le_train_set = lgb.Dataset(le_features, label=train_labels)

# The scikit-learn cross validation api does not include the option for early stopping.
# Perform cross validation with 10 folds
r = lgb.cv(params, oh_train_set, num_boost_round=10000, nfold=10, metrics='auc',
           early_stopping_rounds=100, verbose_eval=False, seed=50)

# Highest score
r_best = np.max(r['auc-mean'])

# Standard deviation of best score
r_best_std = r['auc-stdv'][np.argmax(r['auc-mean'])]

print('The maximium ROC AUC in cross validation was {:.5f} with std of {:.5f}.'.format(r_best, r_best_std))
print('The ideal number of iterations was {}.'.format(np.argmax(r['auc-mean']) + 1))

# Dataframe to hold cv results
random_results = pd.DataFrame(columns=['loss', 'params', 'iteration', 'estimators', 'time', 'ROC AUC'],
                              index=list(range(1, MAX_EVALS)))


def random_objective(params, iteration, n_folds=N_FOLDS):
    """Random search objective function. Takes in hyperparameters
       and returns a list of results to be saved."""

    encoding_type = params['encoding']

    # Handle the encoding
    if encoding_type == 'one_hot':
        train_set = oh_train_set
    elif encoding_type == 'label':
        train_set = le_train_set

    del params['encoding']

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

    params['encoding'] = encoding_type

    # Return list of results
    return [loss, params, iteration, n_estimators, end - start, best_score]


# Iterate through the specified number of evaluations
for i in range(1, MAX_EVALS + 1):

    # Randomly sample parameters for gbm
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}

    # Handle the boosting type
    if params['boosting_type'] == 'goss':

        # Cannot subsample with goss
        params['subsample'] = 1.0
    else:

        # Subsample supported for gdbt and dart
        params['subsample'] = random.sample(subsample_dist, 1)[0]

    # Evaluate the objective function
    results_list = random_objective(params, i)

    if i % 50 == 0:
        # Display the information
        display('Iteration {}: {} Fold CV AUC ROC {:.5f}'.format(i, N_FOLDS, results_list[-1]))

    # Add results to next row in dataframe
    random_results.loc[i, :] = results_list

# Sort results by best validation score
random_results.sort_values('loss', ascending=True, inplace=True)
random_results.reset_index(inplace=True, drop=True)
print(random_results.head())

random_results.to_csv('./random_results_kaggle.csv', index=False)

print(random_results.loc[0, 'params'])

# Find the best parameters and number of estimators
best_random_params = random_results.loc[0, 'params'].copy()
best_random_estimators = int(random_results.loc[0, 'estimators'])
best_random_model = lgb.LGBMClassifier(n_estimators=best_random_estimators, n_jobs=-1,
                                       objective='binary', **best_random_params, random_state=50)

# Fit on the training data
best_random_model.fit(one_hot_features, labels)

# Make test predictions
predictions = best_random_model.predict_proba(one_hot_features_test)[:, 1]

print(
    'The best model from random search scores {:.4f} on the test data.'.format(roc_auc_score(test_labels, predictions)))
print('This was achieved using {} search iterations.'.format(random_results.loc[0, 'iteration']))
