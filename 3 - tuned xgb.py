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
#I performed grid search for obtaining the optimal hyperparameters for XGB
#1 month: learning_rate=0.01, max_depth=3, n_estimators=200, subsample=1
#3 month: learning_rate=0.01, max_depth=3, n_estimators=200, subsample=1
#6 month: learning_rate=0.1, max_depth=3, n_estimators=200, subsample=1

#From that point, I manually tweak the hyperparameters to see whether this improves the balanced accuracy
#With the following source: 
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    
#Evaluated hyperparameters:
#colsample_bytree, gamma, reg_alpha, reg_lambda, min_child_weight


    
#----------------------------------------------------------------------------#

#------------------------------1 MONTH---------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle=False)

#Train
model = xgb.XGBClassifier()
#model = xgb.XGBClassifier(learning_rate=0.01, max_depth=2, n_estimators=250, subsample=1)

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
model = xgb.XGBClassifier(learning_rate=0.01, max_depth=2, n_estimators=500, subsample=1, colsample_bytree=1)

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
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.5, colsample_bytree=1)


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
plt.title('Prediction Accuracies XGB')
plt.legend(loc='lower right')
plt.ylim(0.2, 1.0)
#Show plot
plt.grid(True)
plt.show()

print(cv_scores1.std())
print(cv_scores3.std())
print(cv_scores6.std())