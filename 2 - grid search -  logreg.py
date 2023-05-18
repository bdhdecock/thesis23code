#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:40:34 2023

@author: Basil
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt



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
#Here, I perform the grid search to find the optimal hyperparameters for LRC
"""

X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, shuffle=False)

#Define grid
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['none', 'l2', 'l1', 'elasticnet'], 'tol': [0.01, 0.1, 1], 'solver': 
              ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'fit_intercept': [True, False]}

#Perform grid search to find the optimal hyperparameters
model = LogisticRegression()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='balanced_accuracy')

#Apply GridSearchCV to the data
#Perform on training data
#
grid_search.fit(X_train,y_train)

#Best C and max_iter parameters are displayed in variable Z

z = grid_search.best_params_

#Optimal hyperparameters for 1 month: C = 0.01, fit_intercept = True, penalty = 'none', solver = 'lbfgs', tol = 0.01
#Optimal hyperparameters for 3 month: C = 0.01, fit_intercept = True, penalty = 'none', solver = 'lbfgs', tol = 0.01
#Optimal hyperparameters for 6 month: C = 0.01, fit_intercept = True, penalty = 'none', solver = 'lbfgs', tol = 0.01


#----------------------------------------------------------------------------#
 
"""
#------------------------------1 MONTH---------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle=False)

#Train
model = LogisticRegression(C = 0.01, fit_intercept = True, penalty = 'none', solver = 'lbfgs', tol = 0.1, random_state=1)
model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score1 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores1 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score1 = np.mean(cv_scores1)

#Predict on train
#y_pred_test = model.predict(X_test)

#Balanced accuracy on test
#test_score1 = balanced_accuracy_score(y_test, y_pred_test)


#------------------------------3 MONTH---------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1, shuffle = False)

#Train
model = LogisticRegression(C = 0.01, fit_intercept = True, penalty = 'none', solver = 'lbfgs', tol = 0.01, random_state=1)
model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score3 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores3 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score3 = np.mean(cv_scores3)

#Predict on train
#y_pred_test = model.predict(X_test)

#Balanced accuracy on test
#test_score3 = balanced_accuracy_score(y_test, y_pred_test)


#------------------------------6 MONTH---------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, random_state=1, shuffle=False)

#Train
model = LogisticRegression(C = 0.01, fit_intercept = True, penalty = 'none', solver = 'lbfgs', tol = 0.01, random_state=1)
model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score6 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores6 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score6 = np.mean(cv_scores6)

#Predict on train
#y_pred_test = model.predict(X_test)

#Balanced accuracy on test
#test_score6 = balanced_accuracy_score(y_test, y_pred_test)

#------------------------------PLOT------------------------------------------#
indices = range(1, len(cv_scores1) + 1)

# Plot the accuracies for each array
plt.plot(indices, cv_scores1, marker='o', label='1 month')
plt.plot(indices, cv_scores3, marker='o', label='3 months')
plt.plot(indices, cv_scores6, marker='o', label='6 months')
plt.xlabel('Prediction Iteration')
plt.ylabel('Balanced Accuracy')
plt.title('Prediction Accuracies')

#Legend
plt.legend()

#Show plot
plt.grid(True)
plt.show()


