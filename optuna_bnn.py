import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import os, sys
import optuna
from pathlib import Path
from datetime import datetime
import tensorflow_probability as tfp
tfd = tfp.distributions

from optuna.visualization.matplotlib import plot_optimization_history

class PrintToFile(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
            
out_dir = "./power_spectra/CO/20230313_no_std/"
optuna_dir = "./models/optuna_bnn/CO_20230313_no_std/"
Path(optuna_dir + "plots/").mkdir(parents=True, exist_ok=True)

f = open(optuna_dir + 'logger.txt', 'w')
sys.stdout = PrintToFile(sys.stdout, f)

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, PROGRESS_EPOCH=50):
        self.PROGRESS_EPOCH = PROGRESS_EPOCH
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.PROGRESS_EPOCH == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')}, epoch {epoch}: ", end="")
            for key, val in logs.items():
                print(f"{key}: {val:.3f}", end= "\t")
            print()

print()
print(tf.config.list_physical_devices('GPU'))

df = pd.read_csv('./camels_info/camels_parameters.csv')

def get_LH_files():
    fils = list(map(lambda fil: fil[:-4], \
               (filter(lambda fil: f"LH_" in fil, os.listdir(out_dir)))))
    sorter_1P_fils = lambda filname: int(filname.split('_')[-1].replace('n', '-'))
    return sorted(fils, key=sorter_1P_fils)

sim_names = get_LH_files()
if "LH_603" in sim_names:
    sim_names.remove("LH_603")
    
num_samples = len(sim_names)
print(num_samples)
non_nan_range = np.arange(24, 46)
all_curves = np.zeros((num_samples, 34, 22)) + np.nan
all_cosmologies = np.zeros((num_samples, 6)) + np.nan
for ind, fil in enumerate(sim_names):
    with np.load(out_dir + fil + ".npz", allow_pickle=True) as data:
        curves = data['curves'].item()
        redshifts = data['redshifts'].item()
        ks = data['ks'].item()
    all_curves[ind] = np.array(list(curves.values()))[:, non_nan_range]
    all_cosmologies[ind] = df.loc[df['Name'] == fil].values[0][1:-1]
    
train_split, val_split, test_split = int(0.85*num_samples), \
            int(0.10*num_samples) + 1, int(0.05*num_samples) + 1

print(train_split, val_split, test_split, train_split+val_split+test_split)

train_x, val_x, test_x = np.split(all_curves, [train_split, train_split+val_split])
train_y, val_y, test_y = np.split(all_cosmologies, [train_split, train_split+val_split])

print(train_x.shape, val_x.shape, test_x.shape)
print(train_y.shape, val_y.shape, test_y.shape)

kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                           tf.cast(train_split, dtype=tf.float32))

input_shape = (34, 22, 1) 

# Define loss
def negloglik(y_true, y_pred):
    return -tf.reduce_mean(y_pred.log_prob(y_true))

def objective(trial):
    try:
        n_conv_layers = trial.suggest_int('n_conv_layers', 2, 3)
        n_dense_layers = trial.suggest_int('n_dense_layers', 2, 4)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for i in range(n_conv_layers):
            num_hidden = 2 ** trial.suggest_int(f'log2(n_conv_units_l{i})', 1, 4)
            kernel = trial.suggest_int(f'n_conv_kernel_l{i}', 2, 4)
            model.add(tfp.layers.Convolution2DFlipout(num_hidden, kernel_size=(kernel, kernel),
                  kernel_divergence_fn=kl_divergence_function, activation=tf.nn.tanh))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())

        for i in range(n_dense_layers):
            num_hidden = 2 ** trial.suggest_int(f'log2(n_dense_units_l{i})', 5, 8)
            model.add(tfp.layers.DenseFlipout(num_hidden, activation=tf.nn.tanh,
          kernel_divergence_fn=kl_divergence_function))

        model.add(tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(6)))
        model.add(tfp.layers.MultivariateNormalTriL(6))
    
        model.compile(tf.keras.optimizers.Adadelta(learning_rate=0.2, rho=0.98), \
                      loss=negloglik, # tf.keras.losses.LogCosh(), # ['mse'], 
              metrics=['mae', 'mse'], experimental_run_tf_function=False)

        history = model.fit(train_x, train_y, epochs=5000, verbose=0, callbacks=[CustomCallback(1000)])

        val_loss = model.evaluate(val_x, val_y, verbose=0)
        
        plot_x, plot_y = val_x, val_y
        n_pred = 50
        predictions = np.empty((n_pred, *plot_y.shape))
        for i in range(n_pred):
            predictions[i] = model.predict(plot_x, verbose=0)
        predictions_best = np.nanmean(predictions, axis=0)
        predictions_std = np.nanstd(predictions, axis=0)

        upp_lims = [0.6, 1.1,4.25,4.25,2.25, 2.25]
        low_lims = [0, 0.5, 0, 0, 0.25, 0.25]
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
        fig.subplots_adjust(wspace=0.3, hspace=0.2)
        labels = [r"$\Omega_m$", r"$\sigma_8$", r"$A_{SN1}$", r"$A_{AGN1}$", r"$A_{SN2}$", r"$A_{AGN2}$"]
        for ind, (label, ax, low_lim, upp_lim) in enumerate(zip(labels, axs.ravel(), low_lims, upp_lims)):
            p = np.poly1d(np.polyfit(plot_y[:, ind], predictions_best[:, ind], 1))
            ax.errorbar(plot_y[:, ind], predictions_best[:, ind],  predictions_std[:, ind], \
                        marker="d", ls='none', alpha=0.4)
            ax.set_xlabel("true")
            ax.set_ylabel("prediction")
            ax.plot([low_lim, upp_lim], [low_lim, upp_lim], color="black")
            ax.plot([low_lim, upp_lim], [p(low_lim), p(upp_lim)], color="black", ls=":")
            ax.set_xlim([low_lim, upp_lim])
            ax.set_ylim([low_lim, upp_lim])
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(label)
            ax.grid()
        plt.savefig(optuna_dir + f"plots/{trial.number}.pdf")
        plt.close()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.grid()
        plt.savefig(optuna_dir + f"plots/{trial.number}_hist.pdf")
        plt.close()
        
        return val_loss[0]
    except Exception as e:
        print(e)
        return np.inf
    
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

fig = plot_optimization_history(study)
plt.savefig(optuna_dir + "plots/history.pdf")
plt.close()

print(study.best_params, study.best_value)
