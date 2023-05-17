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

MODEL_FOLDER = 'Models/'

# Create seed
seed_value = 18
seed_value = u.create_seed(seed_value)

# Set information about the model and the training 
vi = 1.366; A = 0.8128; M = 1.2192; B = 1.6256
act = activations.relu
patience_value = 100
nepochs = 800

nn_50 = u.nn([A, M, B], [1, vi, 2], [24, 36, 50], ['U', 'WallDistance'])
nns = [nn_50]

for nn in nns:
    df = u.clean_dataset(nn)
    test_percentage = 0.2
    trn, val = train_test_split(df, test_size=test_percentage, random_state=seed_value, shuffle=True)

    trn_lab = trn.pop('nut')
    val_lab = val.pop('nut')
    if 'U' in nn.inputs:
        trn_U = trn['U']
        val_U = val['U']
    else:
        trn_U = trn.pop('U')
        val_U = val.pop('U')
    nx, ni = trn.shape

    nn.model = u.create_model_norm(nn.layers, trn, act, print_summary=False)
    nn = u.train_and_save_model(nn, patience_value, nepochs, trn, val, trn_lab, val_lab, seed_value, MODEL_FOLDER, save_model=True, plot_hist=True)
    nn, trn_pre, val_pre = u.load_and_predict(nn, trn, val, trn_lab, val_lab, seed_value, MODEL_FOLDER, print_error=True, load_model=True)


    
