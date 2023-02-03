import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import plot_partial_dependence
from sklearn.datasets import make_blobs
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt

# load dataset
df_sa = pd.read_csv("SAheart.data.csv", header = 0)

df_sa = df_sa[list(col for col in df_sa.columns if col != 'famhist')]

np.random.seed(14090)

def get_dataset(task):
  if task == "regression":
    x_train_temp = df_sa[["tobacco", "sbp", "adiposity", "typea", "obesity", "alcohol", "age", "chd"]]
    y_train_temp = df_sa["ldl"]
    feat2index = ["tobacco", "sbp", "adiposity", "typea", "obesity", "alcohol", "age", "chd"]

  elif task == "classification":
    x_train_temp = df_sa[["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]]
    y_train_temp = df_sa["chd"]

    feat2index = ["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]

  x_train_temp.head()

  x_data, y_data = x_train_temp.to_numpy(), y_train_temp.to_numpy().reshape(len(x_train_temp), 1)

  x_train, xx, y_train, yy = train_test_split(x_data, y_data, test_size=0.4)
  x_val, x_test, y_val, y_test = train_test_split(xx, yy, test_size = 0.5)

  return x_train, x_val, x_test, y_train, y_val, y_test, feat2index


def find_split_regression(x, y, feat2index):
  num_data = len(x)
  # j: split variable, s: split value
  # initialize variables of interest 
  j, s, min_loss, min_c = 0, 0, sys.float_info.max, 0
  min_cl, min_cr = sys.float_info.max, sys.float_info.max 
  min_mask = None

  for feat_i, feat in enumerate(feat2index):
    # the algorithm tries every possible integer value of the feature
    for data_i in range(num_data):
      mask = np.full(num_data, False, dtype=bool)
      mask[np.where(x[:, feat_i] < x[data_i, feat_i])] = True 

      l = y[mask]
      r = y[~mask]

      num_l = np.sum(mask == 1)
      num_r = np.sum(mask == 0)

      if num_l == 0 or num_r == 0:
        # print("empty child", l.shape[0], r.shape[0])
        continue

      cl = np.mean(l) 
      cl_val = np.array([cl for _ in range(len(l))])
      cr = np.mean(r)
      cr_val = np.array([cr for _ in range(len(r))])
     
      loss = mean_squared_error(l, cl_val) * (num_l/num_data) + mean_squared_error(r, cr_val) * (num_r/num_data) 
      

      # find the (j,s) pair that gives the lowest c value
      if loss < min_loss:
        min_loss = loss
        min_cl = cl
        min_cr = cr
        j = feat_i
        s = x[data_i, feat_i]
        min_mask = mask

  return {"cl": min_cl, "cr": min_cr, "feature": j, "value": s, "mask": min_mask, "terminal": False}


""""
Find optimal split variable and feature 
"""
def find_split_classification(x, y, feat2index, loss_func):
  num_data = len(x)
  # j: split variable, s: split value
  # initialize variables of interest 
  j, s, min_loss = 0, 0, sys.float_info.max
  min_lk, min_rk = sys.float_info.max, sys.float_info.max
  min_mask = None


  for feat_i, feat in enumerate(feat2index):
    # the algorithm tries every possible integer value of the feature
    for data_i in range(num_data):
      mask = np.full(num_data, False, dtype=bool)
      mask[np.where(x[:, feat_i] < x[data_i, feat_i])] = True 

      l = y[mask]
      
      r = y[~mask]

      # if either region is empty, skip
      if l.shape[0] == 0 or r.shape[0] == 0:
        continue
            
      num_l = np.sum(mask == 1)
      num_r = np.sum(mask == 0)
      
      p_l0 = np.sum(l == 0) / num_l
      p_l1 = np.sum(l == 1) / num_l
      lk = 0 if p_l0 > p_l1 else 1

      p_r0 = np.sum(r == 0) / num_r 
      p_r1 = np.sum(r == 1) / num_r
      rk = 0 if p_r0 > p_r1 else 1
      

      # change loss function
      if loss_func == "misclassification":
        pm_l = max(p_l0, p_l1)
        pm_r = max(p_r0, p_r1)
        loss = (1 - pm_l) * (num_l/num_data) + (1-pm_r)*(num_r/num_data)

      elif loss_func == "gini":
        loss = (1 - (p_l0**2 + p_l1**2))*(num_l/num_data) + (1-(p_r0**2 + p_r1**2)) * (num_r/num_data)
        # print(loss)
        if np.isnan(loss):
          print(p_l0, p_l1, p_r0, p_r1)

      elif loss_func == "crossentropy":
        p_l = max(p_l0, p_l1) - 1e-5
        p_r = max(p_r0, p_r1) - 1e-5

        lnode = -(p_l*np.log(p_l) + (1-p_l) * np.log(1-p_l)) * (num_l/num_data)
        rnode = -(p_r*np.log(p_r) + (1-p_r) * np.log(1-p_r)) * (num_r/num_data)
        loss = lnode + rnode 

        if np.isnan(loss):
          print("isnan", num_l, num_r, p_l, p_r)

      else:
        print("ERROR: Unsupported loss function")

      # find the (j,s) pair that gives the lowest c value
      if loss < min_loss:
        min_loss = loss
        j = feat_i
        s = x[data_i, feat_i]
        min_mask = mask
        min_lk = lk 
        min_rk = rk

  if min_lk == sys.float_info.max:
    print("warning: no split made")

  return {"lk": min_lk, "rk": min_rk, "feature": j, "value": s, "mask": min_mask, "terminal": False}


"""
terminal conditions (stop splitting)
1. min number of data in each node
2. empty node
3. number of terminal nodes equal to max_nodes (tunable parameter)
"""

"""
   root
left  right
"""

def grow_tree_regression(x, y, root, split_info, min_size, max_nodes, feat2index):
  global cur_nodes
  cur_nodes += 2

  indices = split_info["mask"]
  del split_info["mask"]

  left = x[indices]
  right = x[~indices]
  left_y = y[indices]
  right_y = y[~indices]

  root["nums"] = len(x)
  root["feature"] = split_info["feature"]
  root["value"] = split_info["value"]
  root["terminal"] = False

  root["left"] = {"c": split_info["cl"]}
  root["right"] = {"c": split_info["cr"]}
  root["left"]["terminal"] = False
  root["right"]["terminal"] = False


    # Make a node terminal node if it contains less than min_size
  if left.shape[0] < min_size:
    root["left"]["terminal"] = True

  if right.shape[0] < min_size:
    root["right"]["terminal"] = True

  # Define terminal nodes
    # max number of nodes reached
  if cur_nodes >= max_nodes:
    root["left"]["terminal"] = True
    root["right"]["terminal"] = True
    return 

  # Make new splits
  # split left child
  if not root["left"]["terminal"]:
    left_child_info = find_split_regression(left, left_y, feat2index)
    grow_tree_regression(left, left_y, root["left"], left_child_info, min_size, max_nodes, feat2index)

  # split right child
  
  if not root["right"]["terminal"]:
    right_child_info = find_split_regression(right, right_y, feat2index)
    grow_tree_regression(right, right_y, root["right"], right_child_info, min_size, max_nodes, feat2index)

  return


def grow_tree_classification(x, y, root, split_info, min_size, max_nodes, feat2index, loss_func):
  global cur_nodes
  cur_nodes += 2
  
  indices = split_info["mask"]

  del split_info["mask"]

  left = x[indices]
  right = x[~indices]
  left_y = y[indices]
  right_y = y[~indices]


  root["nums"] = len(x)
  root["feature"] = split_info["feature"]
  root["value"] = split_info["value"]
  root["terminal"] = False

  root["left"] = {"k": split_info["lk"]}
  root["right"] = {"k": split_info["rk"]}
  root["left"]["terminal"] = False
  root["right"]["terminal"] = False
  

  # Make a node terminal node if it contains less than min_size
  if left.shape[0] < min_size:
    root["left"]["terminal"] = True

  if right.shape[0] < min_size:
    root["right"]["terminal"] = True



  # Define terminal nodes
    # max number of nodes reached
  if cur_nodes >= max_nodes:
    root["left"]["terminal"] = True
    root["right"]["terminal"] = True
    return 

  # Make new splits
  # split left child
  if not root["left"]["terminal"]:
    left_child_info = find_split_classification(left, left_y, feat2index, loss_func)
    grow_tree_classification(left, left_y, root["left"], left_child_info, min_size, max_nodes, feat2index, loss_func)

  # split right child
  if not root["right"]["terminal"]:
    right_child_info = find_split_classification(right, right_y, feat2index, loss_func)
    grow_tree_classification(right, right_y, root["right"], right_child_info, min_size, max_nodes, feat2index, loss_func)

  return



def build_tree(x, y, min_size, max_nodes, task, feat2index, loss_func):
  if task == "regression":
    split_info = find_split_regression(x, y, feat2index)
    root =  {"c": 0, "feature": split_info["feature"], 
            "value": split_info["value"], "left": {"c": split_info["cl"]}, 
            "right": {"c": split_info["cr"]}, "terminal": False}

    grow_tree_regression(x, y, root, split_info, min_size, max_nodes, feat2index)

  elif task == "classification":
    split_info = find_split_classification(x, y, feat2index, loss_func)
    root =  {"k": -1, "feature": split_info["feature"], 
            "value": split_info["value"], "left": {"k": split_info["lk"]}, 
            "right": {"k": split_info["rk"]}, "terminal": False}

    grow_tree_classification(x, y, root, split_info, min_size, max_nodes, feat2index, loss_func)
  
  else:
    print("ERROR: unsupported task")  
  
  return root



def predict_score(x_test, y_test, root, task):
  y_predict = np.zeros((len(x_test),1))
  for i, row in enumerate(x_test):
    # print(row)
    y_predict[i] = predict_tree(row, root, task)

  if task == "regression":
    error = mean_squared_error(y_predict, y_test)
  elif task == "classification":
    error = np.sum(y_predict == y_test)
    error = len(x_test) - error
  return y_predict, error



def predict_tree(row, root, task):
  while root:
    if root["terminal"]:
      if task == "regression":
        return root["c"]
      elif task == "classification":
        return root["k"]
      else:
        print("ERROR: unsupported task")

    else:
      j, s = root["feature"], root["value"]
      if row[j] < s:
        root = root["left"]
      else:
        root = root["right"]
  
  print("ERROR: no terminal node")



def run_decision_tree_classification(x_trainc, x_valc, y_trainc, y_valc, feat2indexc, losses, max_nodes):
  # get dataset
  task = "classification"


  # hyperparameters
  min_size = 2
  roots = {}

  for j, loss_func in enumerate(losses):
    min_root = {}
    min_error = sys.float_info.max 
    opt_max_node = 0

    for i, max_node in enumerate(max_nodes):
        # global variable
        cur_nodes = 0

        # build decision tree
        root = build_tree(x_trainc, y_trainc, min_size, max_node, task, feat2indexc, loss_func)

        # predict target values
        y_hat, error = predict_score(x_valc, y_valc, root, task)
     

      # compute average error of loss funcs for each max node value
        if error < min_error:
          min_root, opt_max_node = root, max_node
          min_error = error

  # for loss_func in losses:
    roots[loss_func] = [min_root, opt_max_node]



  return roots


# def run_decision_tree_regression(x_train, x_val, y_train, y_val, min_nodes, max_nodes, task, feat2index):
#   # get dataset
#   task = "regression"
#   min_error = sys.float_info.max

#   for max_node in max_nodes:
#     for min_node in min_nodes:
#       print(max_node, min_node)
#       # build decision tree
#       cur_nodes = 0

#       root = build_tree(x_train, y_train, min_node, max_node, "regression", feat2index, None)
#       # predict target values
#       y_hat, reg_error = predict_score(x_val, y_val, root, task)
      
#       print(root)
      
#       if reg_error < min_error:
#         print("error", reg_error)
#         final_root = root
#         min_error = reg_error
#         final_maxnode = max_node
#         final_minnode = min_node

#   return final_root, min_error, final_maxnode, final_minnode



# Cross validation on decision treee regression
max_nodes = [x*10 for x in range(1,5)]
min_nodes = [x*2 for x in range(1,15)]
final_maxnode, final_minnode = 0, 0
final_root = 0
min_error = sys.float_info.max

cur_nodes = 0
task = "regression"
x_train, x_val, x_test, y_train, y_val, y_test, feat2index = get_dataset('regression')
for max_node in max_nodes:
  for min_node in min_nodes:
    # build decision tree
    cur_nodes = 0
    root = build_tree(x_train, y_train, min_node, max_node, task, feat2index, None)
    # predict target values

    y_hat, reg_error = predict_score(x_val, y_val, root, task)
        
    if reg_error < min_error:
      final_root = root
      min_error = reg_error
      final_maxnode = max_node
      final_minnode = min_node

















"""
Regression using decision tree 
"""
y_hat, reg_error = predict_score(x_test, y_test, final_root, "regression")
print("===============================================================")
print("Regression Using Decision Tree")
print(f'MSE of regression: {reg_error}')
print(f'Min max node: {final_maxnode}, Min min node: {final_minnode}')
print("===============================================================")


y = np.mean(y_train)
y = [y] * len(y_test)
error = mean_squared_error(y, y_test)
print("===============================================================")
print("Decision tree regression baseline")
print(f'MSE of decision tree regression using baseline: {error}')
print("===============================================================")

"""
Classification using decision tree
  3 Splitting rules:
    1. Gini index
    2. Misclassification
    3. Crossentropy
"""
task = "classification"
cur_nodes = 0
losses = ["gini", "misclassification", "crossentropy"]
x_trainc, x_valc, x_testc, y_trainc, y_valc, y_testc, feat2indexc = get_dataset("classification")

classification_roots = run_decision_tree_classification(x_trainc, x_valc, y_trainc, y_valc, feat2indexc, losses, [4,5,6,7,8])
for loss_func, root_info in classification_roots.items():
  root = root_info[0]
  # opt_max_node = root_info[1]
  fy_hat, ferror = predict_score(x_testc, y_testc, root, task)
  num_data = len(y_testc)
  # print(f'max node: {opt_max_node}')
  print("===============================================================")
  # print(f'number of 1\'s in y_hat: {np.sum(fy_hat == 1)}')
  # print(f'number of 1\'s in y_test: {np.sum(y_testc == 1)}')
  print(f'# correct predictions using {loss_func}: {num_data - ferror}')
  print(f'% correct predictions using {loss_func}: {(num_data - ferror)/len(x_testc)}')
  print("===============================================================")


y_hat = 1 if np.sum(y_trainc[y_trainc == 1]) > np.sum(y_trainc[y_trainc == 0]) else 0

print("----------TESTING WITH BASELINE----------")
print("===============================================================")
print(f'# correct predictions using baseline: {np.sum(y_hat == y_testc)}')
print(f'% correct predictions using baseline: {np.sum(y_hat == y_testc) / len(y_testc)}')
print("===============================================================")




