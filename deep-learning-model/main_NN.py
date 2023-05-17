import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras import layers, models, activations, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import utilities_NN.py as u

# Create random seed
seed_value = 18
seed_value = u.create_seed(seed_value)

# Set information about the model and the training 
vi = 1.366; A = 0.8128; M = 1.2192; B = 1.6256
patience_value = 100
nepochs = 800

nn_50 = u.nn([24, 36, 50])
nns = [nn_50]

for nn in nns:
    # obtain pandas data-set
    df = u.clean_dataset(nn)
    test_percentage = 0.2
    # split data-set into training and validation set
    trn, val = train_test_split(df, test_size=test_percentage, random_state=seed_value, shuffle=True)
    
    # remove label from training data-set as the eddy viscosity
    trn_lab = trn.pop('nut')
    val_lab = val.pop('nut')
    
    # input tensor shape
    nx, ni = trn.shape
    # create tensorflow model architecture
    nn.model = u.create_model(nn.layers, trn)
    # train and save the model
    nn = u.train_and_save_model(nn, patience_value, nepochs, trn, val, trn_lab, val_lab, seed_value)
    # predict and compute error
    nn, trn_pre, val_pre = u.predict_model(nn, trn, val, trn_lab, val_lab, seed_value)


    
