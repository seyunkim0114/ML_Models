# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F0X-aL0HX1tw0rqRfkwkT4mq89xBbRgO


"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_covtype
from sklearn.inspection import partial_dependence
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

"""## **1. California Housing Data Set**"""

#load data as pandas dataframe
cal = fetch_california_housing()
df = pd.DataFrame(cal.data, columns=cal.feature_names)
df['target'] = pd.Series(cal.target)
df.head()

#https://xgboost.readthedocs.io/en/latest/tutorials/model.html

#check for null values
print(df.isnull().values.any())

#split features and target
X = df[["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]]
y = df["target"]
#X_test = df[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]]
#Y_test = df["median_house_value"]

#check features and target split
print(X.shape)
print(y.shape)

#https://www.kaggle.com/code/joseconomy/california-housing-prices-regression-with-xgboost

#split data into 80% training set and 20% test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#initialize model 
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', learning_rate = 0.1, max_leaf_nodes = 6, n_estimators = 800)

#fit model
xg_reg.fit(X_train,y_train)

#make predictions
y_pred = xg_reg.predict(X_test)

#mse
print(mean_squared_error(y_test, y_pred))

#Graphical representation of predictions and truth
grp = pd.DataFrame({'prediction':y_pred,'Actual':y_test})
grp = grp.reset_index()
grp = grp.drop(['index'],axis=1)
plt.figure(figsize=(20,10))
plt.plot(grp[:120],linewidth=2)
plt.legend(['Actual','Predicted'],prop={'size': 20})

#save AAE
n = range(0, 200)
test_error = []
train_error = []
count = 0
for item in n:
  xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', learning_rate = 0.1, max_leaf_nodes = 6, n_estimators = item)
  xg_reg.fit(X_train,y_train)
  y_pred_test = xg_reg.predict(X_test)
  err = np.sum(abs(y_test - y_pred_test))/y_test.shape[0]
  test_error.append(err)
  y_pred_train = xg_reg.predict(X_train)
  err1 = np.sum(abs(y_train - y_pred_train))/y_train.shape[0]
  train_error.append(err1)
  print(count)
  count+=1

#plot AAE
fig = plt.figure(1, figsize=(10, 5), frameon=False, dpi=100)
fig.add_axes([0, 0, 1, 1])
plt.plot(n, test_error, label = 'test error')
plt.plot(n, train_error, label = 'train error')
plt.xlabel("Iterations M")
plt.ylabel("Absolute Error")
plt.title("Training and Test Absolute Error")
plt.legend()
plt.show()

#relative importance plot
feature_importance = xg_reg.feature_importances_
max = np.max(feature_importance)
relative_importance = []
rel_dict = {}
features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
for item in feature_importance:
  rel = (item/max)*100
  relative_importance.append(rel)
for i in range(0,7):
  rel_dict[features[i]] = relative_importance[i]

rel_dict = sorted(rel_dict.items(), key=lambda x: x[1], reverse=True)
rel_dict = dict(rel_dict)
print(rel_dict)
y = list(rel_dict.values())
x = list(rel_dict.keys())

plt.barh(x, y)
plt.xlabel("Relative Importance")

#partial dependence plots
XGB_v=VotingRegressor([("reg",xg_reg)],).fit(X_train, y_train) #https://stackoverflow.com/questions/62627717/python-sklearn-notfittederror-after-xgboost-fit-has-been-called
feat=["MedInc", "HouseAge", "AveRooms", "AveOccup"]
fig, ax = plt.subplots(figsize=(20, 10))
XGB_RMR=PartialDependenceDisplay.from_estimator(XGB_v, X_train, feat, line_kw={"color": "blue"}, ax=ax)

#https://scikit-learn.org/0.22/auto_examples/inspection/plot_partial_dependence.html
#3D partial dependence plot using 2 features

fig = plt.figure()

XGB_v=VotingRegressor([("reg",xg_reg)],).fit(X_train, y_train)
features = ("AveOccup", "HouseAge")
pdp, axes = partial_dependence(XGB_v, X_train, features=features,
                               grid_resolution=20)
XX, YY = np.meshgrid(axes[1], axes[0])
Z = pdp[0].T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(features[1])
ax.set_ylabel(features[0])
ax.set_zlabel('Partial dependence')
# init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of house value on median\n'
             'age and average occupancy')
plt.subplots_adjust(top=0.9)

plt.show()

"""## **2. Covertype Data Set**"""

#load data as pandas dataframe
#predict forest cover type
cov_type = fetch_covtype()
df = pd.DataFrame(cov_type.data, columns=cov_type.feature_names)
df['target'] = pd.Series(cov_type.target)
df.head()

#check for null values
print(df.isnull().values.any())

#split features and target
X = df[["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]]
y = df["target"]

print(X.shape)
print(y.shape)

#split data into 80% training set and 20% test set
X_train2,X_test2,y_train2,y_test2 = train_test_split(X,y,test_size = 0.2, random_state=0)

#initialize model
xg_class = xgb.XGBClassifier(objective = 'multi:softprob', learning_rate = 0.1, max_leaf_nodes = 6, n_estimators = 400)

#fit model
xg_class.fit(X_train2, y_train2)

#make predictions
y_pred2 = xg_class.predict(X_test2)

#accuracy
print(accuracy_score(y_test2, y_pred2))

#Graphical representation of predictions and truth
grp = pd.DataFrame({'prediction':y_pred2,'Actual':y_test2})
grp = grp.reset_index()
grp = grp.drop(['index'],axis=1)
plt.figure(figsize=(20,10))
plt.plot(grp[:120],linewidth=2)
plt.legend(['Actual','Predicted'],prop={'size': 20})

#save AAE
n = range(0, 20)
test_error2 = []
train_error2 = []
for item in n:
  xg_class2 = xgb.XGBClassifier(objective = 'multi:softprob', learning_rate = 0.1, max_leaf_nodes = 6, n_estimators = item)
  xg_class2.fit(X_train2,y_train2)
  y_pred_test2 = xg_class2.predict(X_test2)
  err = np.sum(abs(y_test2 - y_pred_test2))/y_test2.shape[0]
  test_error2.append(err)
  y_pred_train2 = xg_class2.predict(X_train2)
  err1 = np.sum(abs(y_train2 - y_pred_train2))/y_train2.shape[0]
  train_error2.append(err1)

#plot AAE
fig = plt.figure(1, figsize=(10, 5), frameon=False, dpi=100)
fig.add_axes([0, 0, 1, 1])
plt.plot(n, test_error2, label = 'test error')
plt.plot(n, train_error2, label = 'train error')
plt.xlabel("Iterations M")
plt.ylabel("Absolute Error")
plt.title("Training and Test Absolute Error")
plt.legend()
plt.show()

#plot relative importances
feature_importance = xg_class.feature_importances_
max = np.max(feature_importance)
relative_importance = []
rel_dict = {}
features = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
for item in feature_importance:
  rel = (item/max)*100
  relative_importance.append(rel)
for i in range(0,7):
  rel_dict[features[i]] = relative_importance[i]

rel_dict = sorted(rel_dict.items(), key=lambda x: x[1], reverse=True)
rel_dict = dict(rel_dict)
print(rel_dict)
y = list(rel_dict.values())
x = list(rel_dict.keys())

plt.barh(x, y)
plt.xlabel("Relative Importance")

#partial dependency plots
xg_class2 = xgb.XGBClassifier(objective = 'multi:softprob', learning_rate = 0.1, max_leaf_nodes = 6, n_estimators = 20)
xg_class2.fit(X_train2, y_train2)
fig, ax = plt.subplots(figsize=(20, 10))
features2 = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology"]
XGB_class=PartialDependenceDisplay.from_estimator(xg_class2, X_train2, features2, feature_names = features2, line_kw={"color": "blue"}, ax=ax, target=7)
