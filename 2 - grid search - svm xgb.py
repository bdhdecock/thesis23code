#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 09:39:38 2023

@author: Basil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:18:27 2023

@author: Basil
"""

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from datetime import datetime
import matplotlib.pyplot as plt

np.random.seed(1)

data = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/data_nrm.csv")
y_1month = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_1month.csv")
y_3month = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_3month.csv")
y_6month = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_6month.csv")

X1 = data.drop(data.index[:21])
X3 = data.drop(data.index[:63])
X6 = data.drop(data.index[:126])
y1 = y_1month.drop(y_1month.index[-21:])
y3 = y_3month.drop(y_3month.index[-63:])
y6 = y_6month.drop(y_6month.index[-126:])

#----------------------------------------------------------------------------#
#Here, I perform the grid search to find the optimal hyperparameters for XGB

#Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'loss':['log_loss', 'deviance', 'exponential'],
    'subsample': [ 0.0, 1.0],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [0,2,4],
    'min_samples_leaf': [0,1,2],
    'max_leaf_nodes': [2,4,6],
    'max_features': ['auto', 'sqrt', 'log2'], 
    'max_depth': [2,3,4]}

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle=False)
# Create the XGBoost classifier
xgb_classifier = xgb.XGBClassifier()
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=10, scoring=make_scorer(balanced_accuracy_score))

grid_search.fit(X_train, y_train)
z1 = grid_search.best_params_

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1, shuffle=False)
grid_search.fit(X_train, y_train)
z3 = grid_search.best_params_

X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, random_state=1, shuffle=False)
grid_search.fit(X_train, y_train)
z6 = grid_search.best_params_

#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt



data = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/data_nrm.csv")
y_1month = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_1month.csv")
y_3month = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_3month.csv")
y_6month = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_6month.csv")

#X = data
#y = y_6month
X1 = data.drop(data.index[:21])
X3 = data.drop(data.index[:63])
X6 = data.drop(data.index[:126])
y1 = y_1month.drop(y_1month.index[-21:])
y3 = y_3month.drop(y_3month.index[-63:])
y6 = y_6month.drop(y_6month.index[-126:])



#----------------------------------------------------------------------------#
#In this section, I tune the SVC model by running it on varying values of 
#hyperparameters

#C, Gamma, kernel, degree tuning
#First, I apply GridSearch for C and Gamma separately to improve efficiency
#Perform grid search to estimate the best value of C and gamma


param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale','auto',0.1, 1, 10], 
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
              'degree':[1,2,3,4,5],
              'shrinking':[True,False],
              'probability':[True,False]}
svm = SVC()
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle=False)
grid = GridSearchCV(svm, param_grid, cv=10, scoring='balanced_accuracy')
grid.fit(X_train, y_train)
z1 = grid.best_params_

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1, shuffle=False)
grid.fit(X_train, y_train)
z3 = grid.best_params_

X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, random_state=1, shuffle=False)
grid.fit(X_train, y_train)
z6 = grid.best_params_






