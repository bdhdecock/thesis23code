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
#Here, I perform grid search per model within the voting classifier

#1 month
#For model 1 LR: C = 10, penalty = l1, solver = 'liblinear'
#For model 2 SVC: C = 10, gamma = 'scale', kernel = 'linear'
#For model 3 RF: criterion = 'gini', max_depth=5, min_samples_leaf=2, min_samples_split=10, n_estimators=300

#3 month
#For model 1 LR: C = 10, penalty = 'l1', solver = 'liblinear'
#For model 2 SVC: C = 10, gamma = 'scale', kernel = 'linear'
#For model 3 RF: criterion = 'gini', max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300

#6 month
#For model 1 LR: C = 1, penalty = 'l1', solver = 'liblinear'
#For model 2 SVC: C = 10, gamma = 'scale', kernel = 'rbf'
#For model 3 RF: criterion = 'gini', max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100


#----------------------------------------------------------------------------#

#------------------------------1 MONTH---------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2,shuffle=False, random_state=1)


model1 = LogisticRegression()
model2 = SVC()
model3 = RandomForestClassifier()


#model1 = LogisticRegression( C = 15, penalty = 'l1', solver = 'liblinear')
#model2 = SVC(C = 10, gamma = 'scale', kernel = 'linear')
#model3 = RandomForestClassifier(criterion = 'gini', max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=100)

# Create a voting classifier
model = VotingClassifier(estimators=[('lr', model1), ('svc', model2), ('rf', model3)], voting='hard')

model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score1 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores1 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score1 = np.mean(cv_scores1)


#------------------------------3 MONTH---------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2,shuffle=False, random_state=1)


model1 = LogisticRegression()
model2 = SVC()
model3 = RandomForestClassifier()

#model1 = LogisticRegression(C = 10, penalty = 'l1', solver = 'liblinear')
#model2 = SVC(C = 10, gamma = 'scale', kernel = 'linear')
#model3 = RandomForestClassifier(criterion = 'gini', max_depth=5, min_samples_leaf=1, min_samples_split=2, n_estimators=300)

# Create a voting classifier
model = VotingClassifier(estimators=[('lr', model1), ('svc', model2), ('rf', model3)], voting='hard')

model.fit(X_train, y_train)

#Predict on train
y_pred_train = model.predict(X_train)

#Balanced accuracy on train
train_score3 = balanced_accuracy_score(y_train, y_pred_train)

#CV and average balanced accuracy
cv_scores3 = cross_val_score(model, X_train, y_train, cv=10, scoring='balanced_accuracy')
average_cv_score3 = np.mean(cv_scores3)

#------------------------------6 MONTH---------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2,shuffle=False, random_state=1)

model1 = LogisticRegression()
model2 = SVC()
model3 = RandomForestClassifier()


#model1 = LogisticRegression( C = 1, penalty = 'l1', solver = 'liblinear')
#model2 = SVC(C = 10, gamma = 'scale', kernel = 'rbf')
#model3 = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=300)

# Create a voting classifier
model = VotingClassifier(estimators=[('lr', model1), ('svc', model2), ('rf', model3)], voting='hard')

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
plt.title('Prediction Accuracies Voting')
plt.legend(loc='lower right')
plt.ylim(0.2, 1.0)
#Show plot
plt.grid(True)
plt.show()


print(cv_scores1.std())
print(cv_scores3.std())
print(cv_scores6.std())

