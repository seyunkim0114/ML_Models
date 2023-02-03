# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IrLI7oHSO9b4ezwQwk59hVt4hfGYmubq


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from collections import Counter

def most_find(sequence, n):
    lst = sorted(range(len(sequence)), key=lambda x:sequence[x], reverse=True)
    return lst[:n]

def percent_correct(y_hat, y):
  return sum(y_hat == y) / len(y)

# create fake predictors
x = np.zeros((50, 5000))
for i in range(50):
  x[i,:] = np.random.normal(0, 1, 5000)
print(x)

# create fake labels
y = np.random.randint(2, size=50)

"""1. Done in the wrong way"""

# screen the predictors
corr = np.corrcoef(x, y, rowvar=False)
correlation = corr[:,-1][:-1]

# find the indices of the 100 most significant predictors
corr_ind = most_find(correlation, 100)

x_predictors = x[:, corr_ind]

# perform k-fold cross validation 
scores = []
kf = KFold(n_splits=50)

for train_index, test_index in kf.split(x_predictors):
  x_train, x_test = x_predictors[train_index], x_predictors[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train, y_train)

  y_hat = neigh.predict(x_test)
  score = percent_correct(y_hat, y_test)
  scores.append(score)

# report % correct
print(f'percent correct: {np.mean(scores) * 100}%')
print(f'error rate: {(1-np.mean(scores))*100}%')

"""2. Correct Way

"""

scores = []
kf = KFold(n_splits=50)

for train_index, test_index in kf.split(x):
  corr = np.corrcoef(x[train_index], y[train_index], rowvar=False)
  correlation = corr[:,-1][:-1]

  # find the indices of the 100 most significant predictors
  corr_ind = most_find(correlation, 100)
  x_predictors = x[:, corr_ind]
  x_train, x_test = x_predictors[train_index], x_predictors[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train, y_train)

  y_hat = neigh.predict(x_test)
  score = percent_correct(y_hat, y_test)
  scores.append(score)

# report % correct
print(f'percent correct: {np.mean(scores) * 100}%')
print(f'error rate: {(1-np.mean(scores))*100}%')

""" The error rate for the incorrect way was much lower than for the correct way of doing cross-validation. This is because in the incorrect way, the predictors are chosen on the basis of all the samples meaning it has seen the test set."""