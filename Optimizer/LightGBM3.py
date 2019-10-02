import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# Feature Scaling
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('./data/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# convert our training data into LightGBM dataset format
# (this is mandatory for LightGBM training)
d_train = lgb.Dataset(x_train, label=y_train)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

clf = lgb.train(params, d_train, 150)

# Prediction
y_pred = clf.predict(x_test)
# convert into binary values
for i in range(0, 100):
    if y_pred[i] >= .5:  # setting threshold to .5
        y_pred[i] = 1
    else:
        y_pred[i] = 0

print(y_pred)

cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_pred,y_test)

print(cm)
print(accuracy)
