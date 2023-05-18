#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:13:23 2023

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
                

model = RandomForestClassifier()

#----------------------------------------------------------------------------#
#Here, I perform the grid search to find the optimal hyperparameters for RFC
"""
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, shuffle=False)
param_grid = {
    'n_estimators': [50, 100, 150], 'criterion':['gini', 'entropy', 'log_loss'],
   # 'max_depth': [5, 10, 15, 'none'],
  #  'min_samples_split':[2,3,4],
   # 'min_samples_leaf':[0.0,1,2],
   # 'max_leaf_nodes': [2,4,6,8],
    'bootstrap': [True, False], 
  #  'oob_score': [True, False], 
  #  'n_jobs': ['none', -1, 1, 5],
   # 'max_samples': [True, False],
    'max_features': ['sqrt', 'log2']}
model = RandomForestClassifier(random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='balanced_accuracy', cv=10)
grid_result = grid_search.fit(X_train, y_train)

# Print the best parameters and score
z =  grid_result.best_params_
"""
#1 month: bootstrap = True, criterion = 'gini', max_features = 'sqrt', n_estimators = 50
#3 month: max_depth = 5, max_features = sqrt, n_estimators = 100
#6 month: max_depth = 15, max_features = sqrt, n_estimators = 50
#----------------------------------------------------------------------------#
 
#------------------------------1 MONTH---------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle=False)

model = RandomForestClassifier(bootstrap = True, criterion = 'gini',
                               max_features = 'sqrt',min_samples_leaf=5, n_estimators = 50,
                               max_depth= 3, min_samples_split=2, oob_score=True)
model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score1 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores1 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score1 = np.mean(cv_scores1)


#------------------------------3 MONTH---------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1, shuffle=False)

#Train
model = RandomForestClassifier(bootstrap=True, criterion = 'gini', max_depth=5, max_features='sqrt',
                               min_samples_leaf=5, n_estimators=20,
                               min_samples_split=2, random_state=1, oob_score=True)
model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score3 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores3 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score3 = np.mean(cv_scores3)

#------------------------------6 MONTH---------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, random_state=1, shuffle=False)

#Train
model = RandomForestClassifier(bootstrap=True, criterion = 'gini', max_depth=20, max_features='sqrt',
                               min_samples_leaf=10, n_estimators=10,
                               min_samples_split=2, random_state=1, oob_score=True)
model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score6 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores6 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score6 = np.mean(cv_scores6)

#------------------------------PLOT------------------------------------------#
indices = range(1, len(cv_scores1) + 1)

# Plot the accuracies for each array
plt.plot(indices, cv_scores1, marker='o', label='1 month')
plt.plot(indices, cv_scores3, marker='o', label='3 months')
plt.plot(indices, cv_scores6, marker='o', label='6 months')
plt.xlabel('Prediction Iteration')
plt.ylabel('Balanced Accuracy')
plt.title('Prediction Accuracies Random Forest')
plt.legend(loc='lower right')
plt.ylim(0.2, 1.0)

#Show plot
plt.grid(True)
plt.show()

print(cv_scores1.std())
print(cv_scores3.std())
print(cv_scores6.std())
