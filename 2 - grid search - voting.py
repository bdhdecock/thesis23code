#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:16:29 2023

@author: Basil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:40:05 2023

@author: Basil
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle=False)

#Here, I perform grid search per model within the voting classifier

model1 = LogisticRegression(random_state=1)
model2 = SVC(random_state=1)
model3 = RandomForestClassifier(random_state=1)

# Create a voting classifier
model = VotingClassifier(estimators=[('lr', model1), ('svc', model2), ('rf', model3)], voting='hard')

param_grid1 = {'C': [0.1, 1, 10],
               'penalty': ['l1', 'l2'],
               'solver': ['liblinear', 'saga']}
param_grid2 = {'C': [0.1, 1.0, 10.0],  
               'kernel': ['linear', 'rbf'],  
               'gamma': ['scale', 'auto']}
param_grid3 = {'n_estimators': [100, 200, 300],
               'max_depth': [None, 5, 10],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'criterion': ['gini', 'entropy']}

# Define your individual models and their respective parameter grids
model1 = GridSearchCV(estimator=model1, param_grid=param_grid1, scoring='balanced_accuracy', cv=10)
model2 = GridSearchCV(estimator=model2, param_grid=param_grid2, scoring='balanced_accuracy', cv=10)
model3 = GridSearchCV(estimator=model3, param_grid=param_grid3, scoring='balanced_accuracy', cv=10)

# Fit the grid search to your data
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Get the best parameters
best_params1_1 = model1.best_params_
best_params1_2 = model2.best_params_
best_params1_3 = model3.best_params_



#----------------------------------------------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1, shuffle=False)

#Here, I perform grid search per model within the voting classifier

model1 = LogisticRegression(random_state=1)
model2 = SVC(random_state=1)
model3 = RandomForestClassifier(random_state=1)

# Define your individual models and their respective parameter grids
model1 = GridSearchCV(estimator=model1, param_grid=param_grid1, scoring='balanced_accuracy', cv=10)
model2 = GridSearchCV(estimator=model2, param_grid=param_grid2, scoring='balanced_accuracy', cv=10)
model3 = GridSearchCV(estimator=model3, param_grid=param_grid3, scoring='balanced_accuracy', cv=10)

# Fit the grid search to your data
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Get the best parameters
best_params3_1 = model1.best_params_
best_params3_2 = model2.best_params_
best_params3_3 = model3.best_params_



#----------------------------------------------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1, shuffle=False)

#Here, I perform grid search per model within the voting classifier

model1 = LogisticRegression(random_state=1)
model2 = SVC(random_state=1)
model3 = RandomForestClassifier(random_state=1)

# Define your individual models and their respective parameter grids
model1 = GridSearchCV(estimator=model1, param_grid=param_grid1, scoring='balanced_accuracy', cv=10)
model2 = GridSearchCV(estimator=model2, param_grid=param_grid2, scoring='balanced_accuracy', cv=10)
model3 = GridSearchCV(estimator=model3, param_grid=param_grid3, scoring='balanced_accuracy', cv=10)

# Fit the grid search to your data
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Get the best parameters
best_params3_1 = model1.best_params_
best_params3_2 = model2.best_params_
best_params3_3 = model3.best_params_


#----------------------------------------------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, random_state=1, shuffle=False)

#Here, I perform grid search per model within the voting classifier

model1 = LogisticRegression(random_state=1)
model2 = SVC(random_state=1)
model3 = RandomForestClassifier(random_state=1)

# Define your individual models and their respective parameter grids
model1 = GridSearchCV(estimator=model1, param_grid=param_grid1, scoring='balanced_accuracy', cv=10)
model2 = GridSearchCV(estimator=model2, param_grid=param_grid2, scoring='balanced_accuracy', cv=10)
model3 = GridSearchCV(estimator=model3, param_grid=param_grid3, scoring='balanced_accuracy', cv=10)

# Fit the grid search to your data
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Get the best parameters
best_params6_1 = model1.best_params_
best_params6_2 = model2.best_params_
best_params6_3 = model3.best_params_
