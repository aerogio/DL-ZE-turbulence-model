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
sys.path.append('/NOBACKUP2/gcal/Scripts/Python/')
import general_gio_functions as gio
u = gio.reload_utilities('CF')

MODEL_FOLDER = 'Best_Models/'
# # USE ONLY CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

seed_value = 18
seed_value = u.create_seed(seed_value)
vi = 1.366; A = 0.8128; M = 1.2192; B = 1.6256

# nn7 = nn([M], [vi], [27, 461, 315, 256, 160, 474, 385], ['U', 'WallDistance'])
# nn74i = nn([M], [vi], [27, 461, 315, 256, 160, 474, 385], ['Ux', 'Uy', 'Uz', 'WallDistance'])
# nn75i = nn([M], [vi], [27, 461, 315, 256, 160, 474, 385], ['Ux', 'Uy', 'Uz', 'WallDistance', 'U'])
# nn3 = nn([M], [vi], [9, 175, 414], ['U', 'WallDistance'])
nnOld = u.nn([M], [vi], [64, 64, 32, 16], ['U', 'WallDistance'])
nnT = u.nn([M], [vi], [64, 64, 32, 16], ['U', 'WallDistance', 'Tgrad'])
nnj_512 = u.nn([M], [vi], [25, 264, 214], ['U', 'WallDistance'])
nnj_50 = u.nn([M], [vi], [13, 37, 31], ['U', 'WallDistance'])
nnb_50 = u.nn([M], [vi], [48, 45, 21], ['U', 'WallDistance'])
nnc_50 = u.nn([M], [1, vi, 2], [41, 40, 21], ['U', 'WallDistance'])
nnd_50 = u.nn([M], [1, vi, 2], [13, 37, 31], ['U', 'WallDistance'])
nne_50 = u.nn([A, M, B], [1, vi, 2], [22, 42, 19], ['U', 'WallDistance'])
nnf_50 = u.nn([A, M, B], [1, vi, 2], [24, 36, 50], ['U', 'WallDistance'])
# nn4 = nn([M], [vi], [64, 64, 32, 16], ['Ux', 'Uy', 'Uz', 'WallDistance'])

# nn1 = nn([M], [vi], [299], ['Ux', 'Uy', 'Uz', 'WallDistance'])
# nn0 = nn([M], [vi], [259,450,94,437,385,342], ['U', 'WallDistance'])
# nn1 = nn([M], [vi], [132,15,326,434,377,11,58], ['U', 'WallDistance'])
nn2 = u.nn([M], [vi], [352, 442, 102, 337], ['U', 'WallDistance'])
# nn6 = nn([M], [vi], [243,181,232,67,25,195,147], ['U', 'WallDistance'])
# nn4 = nn([M], [vi], [94, 138, 137, 358], ['Ux', 'Uy', 'Uz', 'WallDistance'])
# nn5 = nn([M], [vi], [92, 101, 159, 450, 313], ['Ux', 'Uy', 'Uz', 'WallDistance'])
# nn6 = nn([M], [vi], [10, 118, 100, 457, 163, 107], ['Ux', 'Uy', 'Uz', 'WallDistance'])
nns = [nnf_50]

# nns = [nn20]


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
    act = activations.relu
    patience_value = 100
    nepochs = 800
    nn.model = u.create_model_norm(nn.layers, trn, act, print_summary=False)
    nn = u.train_and_save_model(nn, patience_value, nepochs, trn, val, trn_lab, val_lab, seed_value, MODEL_FOLDER, save_model=True, plot_hist=True)
    nn, trn_pre, val_pre = u.load_and_predict(nn, trn, val, trn_lab, val_lab, seed_value, MODEL_FOLDER, print_error=True, load_model=True)
    # plot_predictions_midterm(nn, trn_pre, trn_lab, val_pre, val_lab, trn_U, trn['WallDistance'], val_U, val['WallDistance'], 100, plot_ze=False)
    # plot_predictions(nn, trn_pre, trn_lab, val_pre, val_lab, trn_U, trn['WallDistance'], val_U, val['WallDistance'], 100, plot_ze=False)
    u.plot_predictions_paper(nn, trn_pre, trn_lab, val_pre, val_lab, trn_U, trn['WallDistance'], val_U, val['WallDistance'], 50, plot_ze=False)
    # plot_predictions_midterm(nn, trn_pre, trn_lab, val_pre, val_lab, trn_U, trn['WallDistance'], val_U, val['WallDistance'], 100, plot_ze=False)

""" UNCOMMENT TO PLOT HISTORIES """
# plot_histories(nns)

for nn in nns:
    print(nn.model_name)
    print(nn.errors)
plt.show()
print('\nok gio')
