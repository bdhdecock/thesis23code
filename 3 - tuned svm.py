#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:28:47 2023

@author: Basil
"""
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
#Based on the hyperparameter grid search, the following hyperparameters were identified:
#1 month: C=10, gamma='scale', kernel='linear'
#3 month: C=10, gamma='scale', kernel='linear'
#6 month: C=1, gamma = 1, kernel = 'poly'

#After that, manual optimisation is performed in line with the SVC documentation:
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#Evaluated hyperparameters:
#coef0, shrinking, probability, decision_function_shape > no effect

#--------------------------------1 MONTH-------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle=False)

#Train
model = SVC(C=20, gamma='scale', kernel='linear')

#model = SVC(C=20, gamma='scale', kernel='linear')

model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score1 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores1 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score1 = np.mean(cv_scores1)

#--------------------------------3 MONTH-------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1, shuffle=False)

#Train
model = SVC(C=20, gamma='scale', kernel='linear')
model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score3 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores3 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score3 = np.mean(cv_scores3)


#--------------------------------6 MONTH-------------------------------------#
#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, random_state=1, shuffle=False)

#Train
model = SVC(C=0.5, gamma = 0.5, kernel = 'poly')
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
plt.title('Prediction Accuracies SVM')
plt.legend(loc='lower right')
plt.ylim(0.2, 1.0)

#Show plot
plt.grid(True)
plt.show()


print(cv_scores1.std())
print(cv_scores3.std())
print(cv_scores6.std())