#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:24:10 2023

@author: Basil
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

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
#Here, I perform the grid search to find the optimal hyperparameters for DT
#First, test for entropy and gini
"""

param_grid = {'criterion': ['entropy', 'gini', 'log_loss'],'splitter':['best','random'], 'max_depth': [5,10,15,20], 
              'max_leaf_nodes':[5,20,40,60], 'min_samples_split': [2,3,4], 'min_samples_leaf': [5,10,15], 'max_features':['auto','sqrt','log2']}
X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, shuffle=False)
model = DecisionTreeClassifier()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(balanced_accuracy_score))

#Fit on training
grid_search.fit(X_train, y_train)

#Get best parameters
z = grid_search.best_params_
"""
#Optimal parameters
#1 month: criterion='entropy', max_depth=5, max_features='auto', max_leaf_nodes=5, min_samples_leaf=5, min_samples_split=2, splitter='best'
#3 month: criterion='entropy', max_depth=5, max_features='auto', max_leaf_nodes=5, min_samples_leaf=5, min_samples_split=2, splitter='best'
#6 month: criterion='entropy', max_depth=5, max_features='auto', max_leaf_nodes=5, min_samples_leaf=5, min_samples_split=2, splitter='best'
#----------------------------------------------------------------------------#

#------------------------------1 MONTH---------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1)

#Train
model = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_features='auto', max_leaf_nodes=10, min_samples_leaf=8, min_samples_split=2, splitter='best', random_state=1)
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
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, shuffle=False)

#Train
model = DecisionTreeClassifier(criterion='entropy', max_depth=7, max_features='auto', max_leaf_nodes=15, min_samples_leaf=7, min_samples_split=2, splitter='best', random_state=1)
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
X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, shuffle=False)

#Train
model = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_features='auto', max_leaf_nodes=7, min_samples_leaf=3, min_samples_split=2, splitter='best', random_state=1)
model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score6= balanced_accuracy_score(y_train, y_pred_train)

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
plt.ylim(0.2, 1.0)
plt.ylabel('Balanced Accuracy')
plt.title('Prediction Accuracies Decision Trees')
plt.legend(loc='lower right')
plt.ylim(0.2, 1.0)
#Show plot
plt.grid(True)
plt.show()

print(cv_scores1.std())
print(cv_scores3.std())
print(cv_scores6.std())

