import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from contextlib import redirect_stdout
from pathlib import Path

print()
print(tf.config.list_physical_devices('GPU'))

df = pd.read_csv('./camels_info/camels_parameters.csv')

# grid_dir = "20230327_1-2"
grid_dir = "20230403_2_f"
out_dir = f"./grids/CO/{grid_dir}/"
models_dir = f"./models/20230402_CO_{grid_dir}_conv/"
loss = "mse"
# loss = tf.keras.losses.LogCosh()
optimizer = 'adam'

Path(models_dir).mkdir(parents=True, exist_ok=True)

def get_LH_files():
    fils = list(map(lambda fil: fil[:-4], \
               (filter(lambda fil: f"LH_" in fil, os.listdir(out_dir)))))
    sorter_1P_fils = lambda filname: int(filname.split('_')[-1].replace('n', '-'))
    return sorted(fils, key=sorter_1P_fils)

sim_names = get_LH_files()

if "LH_603" in sim_names:
    sim_names.remove("LH_603")
    
print("reading data")
with np.load(out_dir + "all.npz") as data:
    all_curves = data['all_curves']
    all_cosmologies = data['all_cosmologies']
    
num_samples = len(sim_names)
print(num_samples)
print(all_curves.shape, all_cosmologies.shape)

train_split, val_split, test_split = int(0.85*num_samples), \
            int(0.10*num_samples) + 1, int(0.05*num_samples) + 1

print(train_split, val_split, test_split, train_split+val_split+test_split)

train_x, val_x, test_x = np.split(all_curves, [train_split, train_split+val_split])
train_y, val_y, test_y = np.split(all_cosmologies, [train_split, train_split+val_split])

train_x, val_x, test_x = map(lambda arr: np.transpose(arr, axes=[0, 2, 3, 4, 1]), [train_x, val_x, test_x])

print(train_x.shape, val_x.shape, test_x.shape)
print(train_y.shape, val_y.shape, test_y.shape)

# input_shape = (74, 74, 74, 34)
input_shape = (18, 18, 18, 34)
output_num = 6
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
  #   tf.keras.layers.Dense(256, activation='leaky_relu'),
  # tf.keras.layers.Dense(256, activation='leaky_relu'),
  # tf.keras.layers.Dense(256, activation='leaky_relu'),
  tf.keras.layers.Conv3D(256, kernel_size=2, activation='relu'),
  tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
  tf.keras.layers.Conv3D(256, kernel_size=3, activation='relu'),
  # tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
  # tf.keras.layers.Conv3D(256, kernel_size=4, activation='relu'),
  # tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='leaky_relu'),
  tf.keras.layers.Dense(256, activation='leaky_relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(output_num, activation='linear') # assuming 6 output parameters
])

# Compile the model
# model.compile(loss='mse', optimizer='adam')
model.compile(loss=loss, optimizer=optimizer)

model.summary()

with open(models_dir + 'modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
        print(f"loss={loss}")
        print(f"optimizer={optimizer}")

# Train the model
history = model.fit(train_x, train_y, epochs=200, validation_data=(val_x, val_y))

plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'], label="loss", marker="x", ls=":")
plt.plot(history.history['val_loss'], label="val_loss", marker="x", ls=":")
plt.legend()
plt.grid()
plt.yscale("log")
plt.savefig(models_dir + f"/training.pdf")
plt.close()

model.save(models_dir)

predictions = model.predict(val_x)
upp_lims = [0.6, 1.25, 5, 4, 2, 2]
low_lims = [0, 0.5, -1, 0, 0, 0]
labels = ["sigma_m", "omega_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
for ind, (label, ax, low_lim, upp_lim) in enumerate(zip(labels, axs.ravel(), low_lims, upp_lims)):
    p = np.poly1d(np.polyfit(val_y[:, ind], predictions[:, ind], 1))
    ax.scatter(val_y[:, ind], predictions[:, ind])
    ax.plot([low_lim, upp_lim], [low_lim, upp_lim], color="black")
    ax.plot([low_lim, upp_lim], [p(low_lim), p(upp_lim)], color="black", ls=":")
    ax.set_xlim([low_lim, upp_lim])
    ax.set_ylim([low_lim, upp_lim])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(label)
plt.savefig(models_dir + f"/plots.pdf")
plt.close()