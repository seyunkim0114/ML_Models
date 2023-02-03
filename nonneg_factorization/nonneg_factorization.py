# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eb9-QAAr3YDGxwbumGsoKd6GeaINuoOQ
"""

# Seyun Kim, Lucia Rhode, Nishat Ahmed
# Recommendation Model Using Non-negative factorization 

!pip install scikit-surprise

import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import Dataset, NMF
from surprise.model_selection import GridSearchCV
import numpy as np

data_path = "./ml-latest-small"

ratings = pd.read_csv(f'ratings.csv', sep=',')

reader = Reader(rating_scale=(1, 5))

user = ratings["userId"]
movie = ratings["movieId"]
rating = ratings["rating"]

movielens = {
    "user": user,
    "movie": movie,
    "rating": rating
}

df = pd.DataFrame(movielens)

# Loads Pandas dataframe
data = Dataset.load_from_df(df[["user", "movie", "rating"]], reader)

# Loads the builtin Movielens-100k data
movielens = Dataset.load_builtin('ml-100k')

param_grid = {"n_epochs": [5, 10], "n_factors": [5, 10], "biased":[True], 
                "reg_pu":[0.1, 0.06], "reg_qi":[0.06, 0.1]}
gs = GridSearchCV(NMF, param_grid, measures=["rmse", "mae"], cv=4)

gs.fit(data)

# best RMSE score
print(gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print("------Optimal Parameters-------")
print(gs.best_params["rmse"])

algo = gs.best_estimator["rmse"]
algo.fit(movielens.build_full_trainset())
pred = algo.predict(1,1,4)
print(pred)