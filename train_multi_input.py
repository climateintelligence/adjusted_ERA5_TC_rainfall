import os
import numpy as np
import pandas as pd
import argparse
import xarray as xr
import glob
import datetime
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,9)
import models.losses
import models.models
import utils.utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--loss_name", type=str, default="compound", help="Loss function name")
parser.add_argument("--name_tag", type=str, default='MULTI_INPUT', help="String to attach at the beginning of the filename")
args = parser.parse_args()

np.random.seed(args.seed)
tf.random.set_seed(args.seed)
initializer = tf.keras.initializers.GlorotUniform(seed=args.seed)


"""
Setup
"""
# network params
INPUT_SHAPE = (96, 96, 4)
NUM_CLASSES = 1
NUM_STACKS = 3
UNET_FILTERS = [4, 8, 16, 32]
# loss params
LOSS_WEIGHTS = [0.75, 1] # first weight is for FSS/FDS, second for MAE, third for regularization
# loss functions
losses = {'mae': tf.keras.losses.MeanAbsoluteError(),
          'mse': tf.keras.losses.MeanSquaredError(),
          'compound': models.losses.compound_loss_fss(loss_weights=LOSS_WEIGHTS)}

loss_func = losses[args.loss_name]
# save params
SAVE_NAME = f'{args.name_tag}2001_{args.loss_name}_seed_{args.seed}'
SAVE_PATH = f'results/{SAVE_NAME}'
exists = os.path.exists(SAVE_PATH)
if not exists:
    os.mkdir(SAVE_PATH)

# load .csv with information on file names
df = pd.read_csv('data/train_2001_over34/ibtracs_2001_train_over34.csv').sample(frac=1)
# take 80% of data as train, 20% as test
ids = list(df.IDX_TRUE)
ids_train_split, ids_valid_split = train_test_split(ids, test_size=0.2, random_state=42)


"""
Data pipeline
"""
def preprocess_variable(data):
    as_array = np.array(data)
    padded = np.pad(as_array, (2,3), constant_values=0)
    expanded = np.expand_dims(padded, axis=-1)
    return expanded

def load_data(path):
    with xr.open_dataset(path) as data:
        era5_tp = data['tp']
        era5_t = data['t']
        era5_r = data['r']
        era5_tcw = data['tcw']
        mswep_tp = data['precipitation']
        # ERA5 preprocessing
        era5_tp = preprocess_variable(era5_tp)
        era5_t = preprocess_variable(era5_t)
        era5_r = preprocess_variable(era5_r)
        era5_tcw = preprocess_variable(era5_tcw)
        era5_stacked = np.dstack([era5_tp, era5_t, era5_r, era5_tcw])
        # MSWEP preprocessing                
        mswep = preprocess_variable(mswep_tp)
    return era5_stacked, mswep

def train_generator(ids):
    while True:
        for start in range(0, len(ids), args.batch_size):
            x_batch = []
            y_batch = []
            end = min(start + args.batch_size, len(ids))
            ids_train_batch = ids[start:end]
            for idx in ids_train_batch:
                era5, mswep = load_data(f'data/train_2001_over34/merged/merged_{idx}.nc')
                x_batch.append(era5)
                y_batch.append(mswep)
            # form batch and yield
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch


"""
Callbacks
"""
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=6,
                           min_delta=1e-4,
                           verbose=1),
             ReduceLROnPlateau(monitor='val_loss',
                                factor=0.1,
                                patience=4,
                                min_delta=1e-4,
                                verbose=1 ),
             ModelCheckpoint(monitor='val_loss',
                             filepath=f'saved_weights/{SAVE_NAME}.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1),
            CSVLogger('training.csv'),
            TensorBoard(log_dir=log_dir, histogram_freq=1)]


"""
Load and train model
"""
# Load model
model = models.models.arunet_threeblock(input_shape=INPUT_SHAPE,
                                            filters=UNET_FILTERS,
                                            loss_func=loss_func,
                                            initializer=initializer)
model.summary()

# Train model
print(f'Training model: {SAVE_NAME}')
history = model.fit(train_generator(ids=ids_train_split),
                    steps_per_epoch=int(np.ceil(len(ids_train_split) / args.batch_size)),
                    epochs=args.epochs,
                    callbacks=callbacks,
                    validation_data=train_generator(ids_valid_split),
                    validation_steps=int(np.ceil(len(ids_valid_split) / args.batch_size)),
                    verbose=1)


"""
Training plots
"""
# plot MAE
fig, ax = plt.subplots(1, figsize=(14,7))
plt.plot(history.history['mean_absolute_error'], linewidth=2)
plt.plot(history.history['val_mean_absolute_error'], linewidth=2)
plt.title('model MAE', fontsize=26)
plt.ylabel('MAE [mm/3h]', fontsize=26)
plt.xlabel('epoch', fontsize=26)
plt.legend(['train', 'validation'], loc='upper right', fontsize=18)
plt.grid(color='grey', linewidth=0.45)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_xlim(left=0, right=len(history.history['loss'])-1)
plt.savefig(f'{SAVE_PATH}/METRICS.jpg')

# plot loss
fig, ax = plt.subplots(1, figsize=(14,7))
plt.plot(history.history['loss'], linewidth=2)
plt.plot(history.history['val_loss'], linewidth=2)
plt.title('Loss (MAE)', fontsize=26)
plt.ylabel('loss', fontsize=26)
plt.xlabel('epoch', fontsize=26)
plt.legend(['train', 'validation'], loc='upper right', fontsize=18)
plt.grid(color='grey', linewidth=0.45)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_xlim(left=0, right=len(history.history['loss'])-1)
plt.savefig(f'{SAVE_PATH}/LOSS.jpg')



"""
Inference
"""
MODE = 'test'
ERA5_PATH = f'data/{MODE}_2001_over34/ERA5_tp'
MSWEP_PATH = f'data/{MODE}_2001_over34/MSWEP_tp'

mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()
fss = models.losses.fss_loss_filtered

df = pd.read_csv(f'results/{MODE}_2001_metrics_over34.csv')

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Load data
    x_batch = []
    y_batch = []
    IDX = df.loc[idx, 'IDX_TRUE']
    filename = f'data/{MODE}_2001_over34/merged/merged_{IDX}.nc'
    x_batch, y_batch = load_data(filename)
    # Make model prediction
    y_pred = model.predict(x_batch, verbose=0)
#    assert np.sum(y_pred) > 0, 'Empty array!'
    
    # Compute model metrics
    # Pixel-level metrics
    model_mae = mae(y_true=y_batch, y_pred=y_pred).numpy()
    model_mse = mse(y_true=y_batch, y_pred=y_pred).numpy()
    df.loc[idx, f'{SAVE_NAME}_MAE'] = model_mae
    df.loc[idx, f'{SAVE_NAME}_MSE'] = model_mse
    # Spatial accuracy metrics
    model_fss_q80 = fss(y_true=y_batch, y_pred=y_pred, n=15, q=80).numpy()
    model_fss_q95 = fss(y_true=y_batch, y_pred=y_pred, n=15, q=95).numpy()
    model_fss_q99 = fss(y_true=y_batch, y_pred=y_pred, n=15, q=99).numpy()
    df.loc[idx, f'{SAVE_NAME}_FSS_q80'] = model_fss_q80
    df.loc[idx, f'{SAVE_NAME}_FSS_q95'] = model_fss_q95
    df.loc[idx, f'{SAVE_NAME}_FSS_q99'] = model_fss_q99
    # Image-level metrics
    model_corr, _ = pearsonr(y_batch.flatten(), y_pred.flatten())
    model_delta_bias = np.sum(y_batch) - np.sum(y_pred)
    model_abs_delta_bias = np.abs(model_delta_bias)
    model_peak_err = np.abs(np.max(y_batch) - np.max(y_pred))
    df.loc[idx, f'{SAVE_NAME}_corr'] = model_corr
    df.loc[idx, f'{SAVE_NAME}_delta_bias'] = model_delta_bias
    df.loc[idx, f'{SAVE_NAME}_abs_delta_bias'] = model_abs_delta_bias
    df.loc[idx, f'{SAVE_NAME}_peak_err'] = model_peak_err


era5_mean_mae = np.mean(df[f'ERA5_MAE'])
era5_mean_mse = np.mean(df[f'ERA5_MSE'])
era5_mean_fss_q80 = np.mean(df[f'ERA5_FSS_q80'])
era5_mean_fss_q95 = np.mean(df[f'ERA5_FSS_q95'])
era5_mean_fss_q99 = np.mean(df[f'ERA5_FSS_q99'])
era5_mean_corr = np.mean(df[f'ERA5_corr'])
era5_mean_delta_bias = np.mean(df['ERA5_delta_bias'])
era5_mean_abs_delta_bias = np.mean(df['ERA5_abs_delta_bias'])
era5_mean_peak_err = np.mean(df['ERA5_peak_err'])

model_mean_mae = np.mean(df[f'{SAVE_NAME}_MAE'])
model_mean_mse = np.mean(df[f'{SAVE_NAME}_MSE'])
model_mean_fss_q80 = np.mean(df[f'{SAVE_NAME}_FSS_q80'])
model_mean_fss_q95 = np.mean(df[f'{SAVE_NAME}_FSS_q95'])
model_mean_fss_q99 = np.mean(df[f'{SAVE_NAME}_FSS_q99'])
model_mean_corr = np.mean(df[f'{SAVE_NAME}_corr'])
model_mean_delta_bias = np.mean(df[f'{SAVE_NAME}_delta_bias'])
model_mean_abs_delta_bias = np.mean(df[f'{SAVE_NAME}_abs_delta_bias'])
model_mean_peak_err = np.mean(df[f'{SAVE_NAME}_peak_err'])

metrics_list = [SAVE_NAME,
                model_mean_mae,
                model_mean_mse,
                model_mean_fss_q80,
                model_mean_fss_q95,
                model_mean_fss_q99,
                model_mean_corr,
                model_mean_delta_bias,
                model_mean_abs_delta_bias,
                model_mean_peak_err]

metrics_csv = pd.read_csv(f'results/{MODE}_2001_summary_metrics.csv')
metrics_model = pd.DataFrame([metrics_list], columns=metrics_csv.columns.to_list())
metrics_save = pd.concat([metrics_csv, metrics_model])
metrics_save.to_csv(f'results/{MODE}_2001_summary_metrics.csv', index=False)