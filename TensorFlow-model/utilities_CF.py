import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import general_gio_functions as gio
import os
import random
from tensorflow.keras import layers, models, activations, optimizers, losses, callbacks, Sequential, backend
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
import gc
gio.set_font_plot_paper()

# MODEL_FOLDER = 'Models/'
CHECKPOINT_FOLDER = 'Checkpoints/'
HISTORY_FOLDER = 'Histories/'
IMAG_FOLDER = 'Imag'
giocolor = ['#F56AC2','#6CEBC7','#575DD4','#B1D457','#F7BF65']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class nn:
    def __init__(self, sections, velocities, layers, inputs):
        self.sections = sections
        self.velocities = velocities
        self.layers = layers
        self.inputs = inputs
        def get_model_name(self):
            model_name = 'Room'
            for i, section in enumerate(self.sections):
                model_name = model_name + f'-{section}'
            model_name = model_name + '-v'
            for i, velocity in enumerate(self.velocities):
                model_name = model_name + f'-{velocity}'
            model_name = model_name + f'-HL{len(self.layers)}'
            for i, hl in enumerate(self.layers):
                model_name = model_name+f'-{hl}'
            model_name = model_name+f'-I{len(self.inputs)}'
            if 'Tgrad' in self.inputs:
                print('ciao')
                model_name = model_name+'T'
            return model_name
        self.model_name = get_model_name(self)

class nn_opt:
    def __init__(self, sections, velocities, inputs):
        self.sections = sections
        self.velocities = velocities
        self.inputs = inputs
        def get_model_name(self):
            model_name = 'Opt'
            for i, section in enumerate(self.sections):
                model_name = model_name + f'-{section}'
            model_name = model_name + '-v'
            for i, velocity in enumerate(self.velocities):
                model_name = model_name + f'-{velocity}'
            model_name = model_name+f'-I{len(self.inputs)}'
            return model_name
        self.model_name = get_model_name(self)

class trial_nn:
    def __init__(self, nn, trial_id, model, layers):
        self.nn = nn
        self.trial_id = trial_id
        self.model = model
        self.layers = layers
        def get_trial_model_name(self):
            model_name = nn.model_name + f'-Trial-HL_{len(self.layers)}'
            for i, hl in enumerate(self.layers):
                model_name = model_name+f'-{hl}'
            return model_name
        self.model_name = get_trial_model_name(self)

def create_model_opt(trial, trn, act, layers_boundary, nodes_boundary, print_summary=False):
    norm = layers.experimental.preprocessing.Normalization(axis=-1, name='norm')
    norm.adapt(np.array(trn))
    mlp_opt = models.Sequential([norm], name='mlp-opt')
    n_layers = trial.suggest_int("n_layers", layers_boundary[0], layers_boundary[1])
    hidden_nodes = []
    for i in range(n_layers):
        n_node = trial.suggest_int("n_nodes_l{}".format(i), nodes_boundary[0], nodes_boundary[1])
        mlp_opt.add(layers.Dense(n_node, activation=act, name='dense-{}'.format(i)))
        hidden_nodes.append(n_node)
    mlp_opt.add(layers.Dense(1, name='out'))
    
    if print_summary:
        print(mlp_opt.summary())

    return mlp_opt, hidden_nodes

def objective(trial, nn, trn, val, trn_lab, val_lab, act, patience_value, nepochs, layers_boundary, nodes_boundary, seed_value, MODEL_FOLDER):
    model, layers = create_model_opt(trial, trn, act, layers_boundary, nodes_boundary, print_summary=False)
    tnn = trial_nn(nn, trial._trial_id, model, layers)
    print('--> TRIAL # {}'.format(tnn.trial_id))
    tnn = train_and_save_model(tnn, patience_value, nepochs, trn, val, trn_lab, val_lab, seed_value, MODEL_FOLDER, save_model=True, plot_hist=False)
    tnn, trn_pre, val_pre = load_and_predict(tnn, trn, val, trn_lab, val_lab, seed_value, MODEL_FOLDER, print_error=True, load_model=True)
    _ = gc.collect()
    val_error = tnn.errors[1]
    # del tnn.model
    del tnn
    backend.clear_session()
    # plot_predictions(tnn, trn_pre, trn_lab, val_pre, val_lab, trn_U, trn['WallDistance'], val_U, val['WallDistance'], 100, plot_ze=True)
    return val_error

def create_seed(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    return seed_value

def load_Data(filename):
    df = pd.read_csv(filename+'.csv')
    return df

def save_0MLP_clean_to_csv(df, filename):
    df = get_velocity(df)
    DF = df[['U:0', 'U:1', 'U:2', 'yWall','nut', 'U']]
    DF = DF.rename(columns = {'yWall':'WallDistance'})
    DF.to_csv('{}.csv'.format(filename), index=False, float_format='%.6f')
    return

def get_velocity_magnitude(df):
    U_N = (df['Ux']**2+df['Uy']**2+df['Uz']**2)**0.5
    df['U'] = U_N
    return df

def create_clean_dataset(filenames, points):
    DFS = {}
    if len(filenames) > 1:
        for filename in filenames:
            print('--> CREATING CLEAN DATASET \n')
            Df = {}
            for point in points:
                Df[str(point)] = load_Data('Data/'+filename+'-'+str(point))
            if len(points) > 1:
                i = 0
                DF = Df[str(points[i])]
                for i in range(len(points)-1):
                    i = i + 1 
                    DF = DF.append(Df[str(points[i])], ignore_index=True)
            else:
                DF = Df[str(points[0])]
            save_0MLP_clean_to_csv(DF, 'Clean-Data/'+filename)
            df = load_Data('Clean-Data/'+filename+'-clean')
            print('--> DATASET LOADED \n')
            DFS[filename] = df
        i = 0
        df = DFS[filenames[i]]
        for i in range(len(filenames)-1):
            i = i+1
            # df = df.append(DFS[filenames[i]], ignore_index=True)
            df = pd.concat([df, DFS[filenames[i]]])
        df.to_csv('Clean-Data/data-total-clean.csv', index=False, float_format='%.6f')
    return df

def clean_dataset(nn):
    print('\n--> CLEANING DATASET')
    cols = nn.inputs + ['nut']
    DF = pd.DataFrame(columns=cols)
    for velocity in nn.velocities:
        for section in nn.sections:
            df = load_Data('Data/d-'+str(velocity)+'-'+str(section))
            df = df.rename(columns = {'yWall':'WallDistance', 'U:0':'Ux','U:1':'Uy','U:2':'Uz'})
            df = get_velocity_magnitude(df)
            if 'Tgrad' in cols:
                Tgrad = (df['grad(T):0']**2+df['grad(T):1']**2+df['grad(T):2']**2)**0.5
                df['Tgrad'] = Tgrad
            df = df[cols]
            if 'Ux' in cols:
                df = get_velocity_magnitude(df)

                
            DF = pd.concat([DF, df], ignore_index=True)

    return DF

def get_dataset(df, exp, train_percentage, label, seed_value, plot_data=False):
    df1 = df.sample(frac=exp, random_state=seed_value)
    train_dataset = df1.sample(frac=train_percentage, random_state = seed_value)
    test_dataset = df.drop(train_dataset.index)
    if plot_data:
        sns.pairplot(train_dataset, diag_kind='kde')
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop(label)
    test_labels = test_features.pop(label)
    return train_features, test_features, train_labels, test_labels

def create_model_norm(hidden_layers, trn, act, print_summary=False):
    norm = layers.experimental.preprocessing.Normalization(axis=-1, name='norm')
    norm.adapt(np.array(trn))
    mlp = models.Sequential([norm], name='mlp')
    for i, n_nodes in enumerate(hidden_layers):
        mlp.add(layers.Dense(n_nodes, activation=act, name='dense-{}'.format(i)))
    mlp.add(layers.Dense(1, name='out'))
    if print_summary:
        print(mlp.summary())
    return mlp

def train_and_save_model(nn, patience_value, nepochs, trn, val, trn_out, val_out, seed_value, MODEL_FOLDER, save_model=False, plot_hist=False):
    if os.path.exists(MODEL_FOLDER+nn.model_name):
        print('\n--> Model {} is trained already, skipping training'.format(nn.model_name))
        nn.history = pd.read_csv(HISTORY_FOLDER+'CSV/'+nn.model_name+'.csv')
        
    else:
        print('\n--> TRAINING '+nn.model_name+' \n')
        Es = callbacks.EarlyStopping(monitor='val_loss', patience=patience_value)
        
        CHECKPOINT_FILEPATH = '{}cp_{}'.format(CHECKPOINT_FOLDER, nn.model_name)
        checkpoint = callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_FILEPATH,
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
        nn.model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        history = nn.model.fit(trn, trn_out, epochs=nepochs, callbacks=callbacks_list, validation_data=(val, val_out), verbose=2)
        nn.history = history.history
        # if len(history.history['loss']) =
        backend.clear_session()
        # save the trained model
        if save_model:
            
            nn.model.save(MODEL_FOLDER+nn.model_name)
            print('--> MODEL SAVED \n')
            history_df = pd.DataFrame(nn.history)
            history_df.to_csv('{}CSV/{}.csv'.format(HISTORY_FOLDER, nn.model_name))
    if len(nn.history['loss'])==nepochs:
        print(f"{bcolors.FAIL}Warning: Reached Max number of epochs.\nConsider increasing the number of epochs for better training.{bcolors.ENDC}")
        nn.z = 0
    else:
        nn.z = 1
    if plot_hist:
        plot_history(nn.history, nn.model_name, patience_value)

    return nn

def load_and_predict(nn, trn, val, trn_lab, val_lab, seed_value, MODEL_FOLDER, print_error=False, load_model=False):
    print('-->LOADING {}'.format(nn.model_name))
    if load_model:
        nn.model = models.load_model(MODEL_FOLDER+nn.model_name)
    seed_value = create_seed(seed_value)
    trn_pre = nn.model.predict(trn)
    val_pre = nn.model.predict(val)
    trn_rmse, trn_nrmse = calculate_RMSE(trn_pre, trn_lab)
    val_rmse, val_nrmse = calculate_RMSE(val_pre, val_lab)    
    if print_error:
        print('\nTraining NRMSE is {:.3f} %'.format(trn_nrmse))
        print('Validation NRMSE is {:.3f} %'.format(val_nrmse))
    nn.errors = [trn_nrmse, val_nrmse]
    backend.clear_session()
    return nn, trn_pre, val_pre
        

def plot_predictions(nn, trn_pre, trn_lab, val_pre, val_lab, trn_U, WallDistance, val_U, val_WallDistance, n, plot_ze=False):
    x = range(n)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize= (10, 6))
    ax[0].plot(x, trn_lab.head(n), color='black', linewidth = 1.7, label=r'RNG $k-\varepsilon$ $\nu_t$')
    ax[0].plot(x, trn_pre[:n], '-.', alpha=0.7, label = 'NN Predictions')
    ax[1].plot(x, val_lab.head(n), color='black', linewidth = 1.7, label=r'RNG $k-\varepsilon$ $\nu_t$')
    ax[1].plot(x, val_pre[:n], '-.', alpha=0.7, label = 'NN Predictions')
    if plot_ze:
        nut_ze = 0.03874*trn_U*WallDistance
        val_nut_ze = 0.03874*val_U*val_WallDistance
        ax[0].plot(x, nut_ze.head(n), '--',linewidth = 1, alpha=0.7, color='green',label=r'Zero-equation $\nu_t$')
        ax[1].plot(x, val_nut_ze.head(n), '--',linewidth = 1, alpha=0.7, color='green',label=r'Zero-equation $\nu_t$')
        ze_rmse, ze_nrmse = calculate_RMSE(trn_lab, nut_ze)
        val_ze_rmse, val_ze_nrmse = calculate_RMSE(val_lab, val_nut_ze)
        print('The algebraic zero equation model trn NRMSE is {:.3f} %'.format(ze_nrmse))
        # print('The algebraic zero equation model val NRMSE is {:.3f} %'.format(val_ze_nrmse))
    ax[1].set_xlabel('dataset')
    ax[0].set_ylabel(r'$\nu_t$')
    ax[1].set_ylabel(r'$\nu_t$')
    ax[1].grid(color='0.9')
    ax[0].grid(color='0.9')
    ax[0].set_title('First {}'.format(n)+' train predictions')
    ax[1].set_title('First {}'.format(n)+' test predictions')
    plt.legend()
    fig.suptitle(nn.model_name)
    plt.savefig('{}/P-{}.png'.format(IMAG_FOLDER, nn.model_name), transparent=True)

    return


def plot_predictions_paper(nn, trn_pre, trn_lab, val_pre, val_lab, trn_U, WallDistance, val_U, val_WallDistance, n, plot_ze=False):
    x = range(n)
    fig, ax = plt.subplots(1, 1, sharex=True, figsize= (7, 4))
    ax.plot(x, trn_lab.head(n), color="#2C61DB", linewidth=1.7, marker='.',  markersize=10, label=r'RNG $k-\varepsilon$ - $\nu_t$', zorder=20)
    # ax.scatter(x, trn_lab.head(n), color="#2C61DB", label=r'RNG $k-\varepsilon$ $\nu_t$', zorder=10)

    ax.plot(x, trn_pre[:n], '--', alpha=1, linewidth=1.7, label = 'MLP predictions', color="#DB3D1D", zorder=15)
    # ax.scatter(x, trn_pre[:n], alpha=1,  label = 'NN Predictions', color="#DB3D1D")
    
    if plot_ze:
        nut_ze = 0.03874*trn_U*WallDistance
        val_nut_ze = 0.03874*val_U*val_WallDistance
        ax.plot(x, nut_ze.head(n), '--',linewidth = 1, alpha=0.7, color='green',label=r'Zero-equation $\nu_t$')
        ze_rmse, ze_nrmse = calculate_RMSE(trn_lab, nut_ze)
        val_ze_rmse, val_ze_nrmse = calculate_RMSE(val_lab, val_nut_ze)
        print('The algebraic zero equation model trn NRMSE is {:.3f} %'.format(ze_nrmse))
        # print('The algebraic zero equation model val NRMSE is {:.3f} %'.format(val_ze_nrmse))

    ax.set_ylabel(r'$\nu_t$')
    ax.set_xlabel(r'predictions')
    ax.grid(color='0.9')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(3,1))
    fig.tight_layout()
    # ax.set_title('First {}'.format(n)+' train predictions')
    plt.legend()
    # fig.suptitle(nn.model_name)
    plt.savefig('{}/P-{}.png'.format(IMAG_FOLDER, nn.model_name), transparent=True)
    plt.savefig('{}/P-{}.eps'.format(IMAG_FOLDER, nn.model_name), transparent=True)

    return


def plot_predictions_midterm(nn, trn_pre, trn_lab, val_pre, val_lab, trn_U, WallDistance, val_U, val_WallDistance, n, plot_ze=False):
    x = range(n)
    fig, ax = plt.subplots(1, 1, sharex=True, figsize= (8, 2), squeeze=False)
    ax = ax.ravel()    
    ax[0].plot(x, val_lab.head(n), color='black', linewidth = 1.5, label=r'$\nu_t$ label')
    ax[0].plot(x, val_pre[:n], '-.', alpha=1, color='#DB3D1D', label = 'Predictions')
    if plot_ze:
        nut_ze = 0.03874*trn_U*WallDistance
        val_nut_ze = 0.03874*val_U*val_WallDistance
        # ax[0].plot(x, nut_ze.head(n), '--',linewidth = 1, alpha=0.7, color='green',label=r'Zero-equation $\nu_t$')
        ax[1].plot(x, val_nut_ze.head(n), '--',linewidth = 1, alpha=0.7, color='green',label=r'Zero-equation $\nu_t$')
        ze_rmse, ze_nrmse = calculate_RMSE(trn_lab, nut_ze)
        val_ze_rmse, val_ze_nrmse = calculate_RMSE(val_lab, val_nut_ze)
        print('The algebraic zero equation model trn NRMSE is {:.3f} %'.format(ze_nrmse))
        # print('The algebraic zero equation model val NRMSE is {:.3f} %'.format(val_ze_nrmse))
    ax[0].set_xlabel('dataset')
    # ax[0].set_ylabel(r'$\nu_t$')
    ax[0].set_ylabel(r'$\nu_t$')
    ax[0].grid(color='0.9')
    # ax[0].grid(color='0.9')
    # ax[1].set_title('First {}'.format(n)+' test predictions')
    plt.legend(frameon=False)
    # fig.suptitle(nn.model_name)
    ax[0].axis('off')
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([])
    fig.tight_layout()
    fig.savefig('/NOBACKUP2/gcal/Dropbox/PhD/tmp/imag/midterm-{}.png'.format(nn.model_name), transparent=True)

    return


def plot_predictions_no_norm(nn, pre, lab, title, trn_U, WallDistance, df_mean, n, plot_ze=False):
    plt.figure()
    plt.grid(color= '0.9')
    x = range(n)
    plt.plot(x, lab.head(n)+df_mean['nut'], color='black', linewidth = 1.7, label=r'RNG $k-\varepsilon$ $\nu_t$')
    plt.plot(x, pre[:n]+df_mean['nut'], '-.', alpha=0.7, label = '{} Predictions'.format(nn.model_name))
    if plot_ze:
        WallDistance = WallDistance + df_mean['WallDistance']
        trn_U = trn_U + df_mean['U']
        nut_ze = 0.03874*trn_U*WallDistance
        plt.plot(x, nut_ze.head(n), '--',linewidth = 1, alpha=0.7, color='green',label=r'Zero-equation $\nu_t$')
        ze_rmse, ze_nrmse = calculate_RMSE(lab, nut_ze-df_mean['nut'])
        print('The algebraic zero equation model NRMSE is {:.3f} %'.format(ze_nrmse))        
    plt.margins(y=0.5)
    plt.xlabel('dataset')
    plt.ylabel(r'$\nu_t$')
    plt.title('First {}'.format(n)+title+' predictions')
    plt.legend()

    return

def plot_history(hist, model_name, patience_value):    
    plt.figure()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(hist['loss'], label='train', color='black', zorder=100)
    ax.plot(hist['val_loss'], '--', linewidth=0.8, alpha=0.8, color='grey', label='validation', zorder=100)
    # plt.title('{} loss'.format(model_name))
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_yscale('log')
    epochs = len(hist['loss'])-1
    # ax.fill_between([epochs-patience_value, epochs], [0, 0], [1,1], color='0.9', alpha=0.5, zorder=10)
    ax.scatter(np.argmin(hist['val_loss']), min(hist['val_loss']), color='red', zorder=1000, label='min val loss')

    ax.set_ylim(top=max(hist['loss']))
    # plt.xscale('log')
    fig.tight_layout()
    ax.legend(loc='upper center')
    ax.grid(which='both', color='0.9', zorder =-10)
    ax.grid(True,color='0.75',axis='y',linewidth=1.3, zorder=0)
    plt.savefig('{}Imag/hist_{}.png'.format(HISTORY_FOLDER, model_name), transparent=True)
    plt.savefig('{}Imag/hist_{}.eps'.format(HISTORY_FOLDER, model_name), transparent=True)
    return

def plot_histories(nns):
    plt.figure()
    for i, nn in enumerate(nns):
        plt.plot(nn.history['val_loss'], label='val_loss {}'.format(nn.model_name), color=giocolor[i])
        plt.scatter(np.argmin(nn.history['val_loss']), min(nn.history['val_loss']), s=35, color=giocolor[i], zorder=1000, label='min val loss {}'.format(nn.model_name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend(loc='best')

    plt.grid(which='both', color='0.9')
    plt.grid(True,color='0.75',axis='y',linewidth=1.3)
    plt.title('Validation losses')
      
    return

def calculate_RMSE(pre, lab):
    pre = np.reshape(pre, (-1,))
    lab = np.reshape(lab, (-1,))
    rmse = mean_squared_error(lab, pre, squared=False)
    nrmse = rmse*100 / (max(lab)-min(lab))
    return rmse, nrmse                


def plot_optuna_plots(study):
    plot_optimization_history(study)
    plot_parallel_coordinate(study, params=["n_layers"])
    plot_parallel_coordinate(study, params=["n_nodes_l0"] )
    # plot_contour(study, params=["n_layers"])
    plot_contour(study)
    plot_slice(study)
    plot_param_importances(study)
    plot_param_importances(
        study, target=lambda t: t.duration.total_seconds(), target_name="duration")
    plot_edf(study)
    return
