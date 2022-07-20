# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:40:03 2021

@author: explo
"""

import numpy as np

from sklearn.metrics import mean_squared_error
from deepforest import CascadeForestRegressor

# load CAM5 test set and training sets with %ages of high-resolution data
TestSet = np.load('CESM Dataset/multi-resolution/Split Datasets/Test Set.npy')
TrainingSet_00 = np.load('CESM Dataset/multi-resolution/Split Datasets/baseTrainingSet.npy')
TrainingSet_05 = np.load('CESM Dataset/multi-resolution/Split Datasets/High Resolution Training Sets/TrainingSet_05.npy')
TrainingSet_20 = np.load('CESM Dataset/multi-resolution/Split Datasets/High Resolution Training Sets/TrainingSet_20.npy')
TrainingSet_50 = np.load('CESM Dataset/multi-resolution/Split Datasets/High Resolution Training Sets/TrainingSet_50.npy')
TrainingSet_100 = np.load('CESM Dataset/multi-resolution/Split Datasets/High Resolution Training Sets/TrainingSet_100.npy')


# functions for splitting x and y data
def splitSetEB(array):
    X = array[:, 0:10] # select first 11 columns (training)
    Y = array[:, [11]] # select column 12 (energy balance)
    return(X, Y)

def splitSetP(array):
    X = array[:, 0:10] # select first 11 columns (training)
    Y = array[:, [12]] # select column 13 (precipitation)
    return(X, Y)
    

X_Test, Y_Test = splitSetEB(TestSet) # initialize eval sets

model = CascadeForestRegressor(# initialize with params
    n_bins=150, 
    bin_subsample=200000, 
    bin_type='percentile', 
    max_layers=20, 
    criterion='mse', 
    n_estimators=5, 
    n_trees=550, 
    max_depth=None, 
    min_samples_leaf=2, 
    use_predictor=True, 
    predictor='forest', 
    predictor_kwargs={}, 
    backend='custom', 
    n_tolerant_rounds=2, 
    delta=1e-05, 
    partial_mode=False, 
    n_jobs=-1, 
    random_state=1, 
    verbose=1
    )      

evalSet = TrainingSet_100  # assign which set to evaluate

# train model on base set combined with 5% of high-resolution set
X_train, Y_train = splitSetEB(evalSet)       # split training sets
model.fit(X_train, np.ravel(Y_train))        # train the model
Y_Pred = model.predict(X_Test)               # predict based on test set
mse = mean_squared_error(Y_Test, Y_Pred)     # mse between model output and test set
print("\nTesting MSE: {:.3f}".format(mse))   # print

