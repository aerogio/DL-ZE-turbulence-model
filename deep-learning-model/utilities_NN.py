import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras import layers, models, activations, optimizers, losses, callbacks, Sequential, backend
from sklearn.metrics import mean_squared_error

class nn:
    """ Class to define a NN based on the amount of hidden layers and nodes per layer
    layers - list indicating amount of nodes in each hidden layer
    """
    def __init__(self, layers):
        self.layers = layers
        def get_model_name(self):
            model_name = model_name + f'HL{len(self.layers)}'
            for i, hl in enumerate(self.layers):
                model_name = model_name+f'-{hl}'
            return model_name
        self.model_name = get_model_name(self)

def create_seed(seed_value):
    """ Create random seed """
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    return seed_value

def load_Data(filename):
    df = pd.read_csv(filename+'.csv')
    return df

def get_velocity_magnitude(df):
    U_N = (df['Ux']**2+df['Uy']**2+df['Uz']**2)**0.5
    df['U'] = U_N
    return df

def clean_dataset(nn):
    """ Load data from Data/  
    INPUT
    nn: neural network class
    OUTPUT
    DF: pandas data-set
    """
    print('\n--> CLEANING DATASET')
    cols = ['U', 'WallDistance', 'nut']
    DF = pd.DataFrame(columns=cols)
    # for each file of data 
    for filename in list-of-fata:
        df = load_Data('Data/filename'))
        df = df.rename(columns = {'yWall':'WallDistance', 'U:0':'Ux','U:1':'Uy','U:2':'Uz'})
        df = get_velocity_magnitude(df)
        df = df[cols]       
    DF = pd.concat([DF, df], ignore_index=True)
    return DF

def create_model(hidden_layers, trn):
    """ Create model architectures:
    INPUT
    hidden_layers: list with layers and nodes per layer
    trn: training dataset
    
    OUTPUT
    NN: neural network model
    """
    
    # preprocessing normalization layer
    norm = layers.experimental.preprocessing.Normalization(axis=-1, name='norm')
    norm.adapt(np.array(trn))
    NN = models.Sequential([norm], name='NN')
    
    for i, n_nodes in enumerate(hidden_layers):
        NN.add(layers.Dense(n_nodes, activation=activations.relu, name='dense-{}'.format(i)))
    NN.add(layers.Dense(1, name='out'))
    return NN

def train_and_save_model(nn, patience_value, nepochs, trn, val, trn_lab, val_lab, seed_value):
    """ Function to train and save the deep learning model
    INPUT
    nn: neural newtork class
    patience_value: value to check for early stopping
    nepochs: maximum epoch number
    trn, val: training/validating input
    trn_lab, val_lab: training/validating label"
    seed_value
    OUTPUT
    nn: trained neural network class
    """
    print('\n--> TRAINING '+nn.model_name+' \n')
    # define checkpoint for early stopping
    Es = callbacks.EarlyStopping(monitor='val_loss', patience=patience_value)   
    Checkpoint_filepath = f'Checkpoints/cp_{nn.model_name}'
    checkpoint = callbacks.ModelCheckpoint(
            filepath=Checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
    callbacks_list = [Es, checkpoint]
        
    step = tf.Variable(0, trainable=False)
    boundaries = [100, 100]
    values = [1e-3, 1e-4, 1e-5]
    lr = optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    opt = optimizers.Adam(learning_rate=lr(step))
    seed_value = create_seed(seed_value)
    
    # compile model
    nn.model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    # train model
    history = nn.model.fit(trn, trn_out, epochs=nepochs, callbacks=callbacks_list, validation_data=(val, val_out), verbose=2)
    nn.history = history.history
    backend.clear_session()
    nn.model.save(f'Models/{nn.model_name}')
    print('--> MODEL SAVED \n')
    return nn

def predict_model(nn, trn, val, trn_lab, val_lab, seed_value):
    """ Function to predict the output of the neural network
    INPUT
    nn: neural network class
    trn, val: training/validating input
    trn_lab, val_lab: training/validating labels
    seed_value
    OUTPUT
    nn: updated neural network class with errors
    trn_pre, val_pre: prediction on the training and validating data-set 
    """
    seed_value = create_seed(seed_value)
    # make prediction
    trn_pre = nn.model.predict(trn)
    val_pre = nn.model.predict(val)
    # calculate error
    trn_rmse, trn_nrmse = calculate_RMSE(trn_pre, trn_lab)
    val_rmse, val_nrmse = calculate_RMSE(val_pre, val_lab)    
    nn.errors = [trn_nrmse, val_nrmse]
    backend.clear_session()
    return nn, trn_pre, val_pre

def calculate_RMSE(pre, lab):
    """ Function to calculate root mean squared error between label and prediction
    INPUT
    pre, lab: prediction/label
    OUTPUT
    rmse: root mean squared error
    nrmse: normalized root mean squared error
    """
    pre = np.reshape(pre, (-1,))
    lab = np.reshape(lab, (-1,))
    rmse = mean_squared_error(lab, pre, squared=False)
    nrmse = rmse*100 / (max(lab)-min(lab))
    return rmse, nrmse                
