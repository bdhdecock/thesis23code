#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:20:10 2023

@author: Basil
"""

#ANN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, SGD



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



param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Different sizes for hidden layers
    'activation': ['logistic', 'relu'],  # Activation functions to try
    'solver': ['adam', 'sgd'],  # Solvers to use
    'learning_rate': ['constant', 'adaptive'],  # Learning rate types
    'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rates
    'max_iter': [100, 200, 300],  # Maximum number of iterations
    'alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
    'batch_size': [16, 32, 64]  # Batch sizes for training
}

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle=False)
#Fit to the data
model = MLPClassifier()
grid_search = GridSearchCV(model, param_grid, cv=10, scoring=make_scorer(balanced_accuracy_score))
grid_search.fit(X_train, y_train)
z1 = grid_search.best_params_

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=1, shuffle=False)
grid_search = GridSearchCV(model, param_grid, cv=10, scoring=make_scorer(balanced_accuracy_score))
grid_search.fit(X_train, y_train)
z3 = grid_search.best_params_

X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size=0.2, random_state=1, shuffle=False)
grid_search = GridSearchCV(model, param_grid, cv=10, scoring=make_scorer(balanced_accuracy_score))
grid_search.fit(X_train, y_train)
z6 = grid_search.best_params_



