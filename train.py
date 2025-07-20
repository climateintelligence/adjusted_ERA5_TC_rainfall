import os
import numpy as np
import pandas as pd
import argparse
import xarray as xr
import glob
import datetime
from scipy.stats import pearsonr, variation
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import models.losses
import models.models
import utils.utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import time

"""
Parser
"""
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=120, help="Number of epochs for training")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--n", type=int, default=15, help="Size of patch over which to calculate FSS")
parser.add_argument("--loss_name", type=str, default="compound", help="Loss function name")
parser.add_argument("--mult_factor", type=int, default=8, help="Factor by which to multiply the filters")
parser.add_argument("--alpha", type=float, default=0.5, help="Controls the scaling of the loss components")
parser.add_argument("--lambda_", type=float, default=1, help="Controls the steepness of my exponential weighting")
parser.add_argument("--kappa", type=float, default=0.1, help="Controls the contribution of the error to my weighting")
parser.add_argument("--num_blocks", type=int, default=3, help="Number of UNet blocks in each branch")
parser.add_argument("--delta", type=float, default=3.0, help="Delta value for the Huber loss")
parser.add_argument("--peak_weight", type=float, default=0.05, help="Weight of the peak error component")
parser.add_argument("--name_tag", type=str, default='multiinput', help="String to attach at the beginning of the filename")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Initial value of Adam's learning rate")
args = parser.parse_args()

start = time.time()
"""
Setup
"""
# Seeds
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
initializer = tf.keras.initializers.GlorotUniform(seed=args.seed)

# Network params
INPUT_SHAPE = (96, 96, 4)
UNET_FILTERS = (4, 8, 16, 32, 64)
UNET_FILTERS = [element * args.mult_factor for element in UNET_FILTERS]

# Loss params
LOSS_WEIGHTS = (0.75, 1, args.peak_weight) # first weight is for FSS/FDS, second for MAE, third for regularization

# Loss functions
losses = {'mae': tf.keras.losses.MeanAbsoluteError(),
          'mse': tf.keras.losses.MeanSquaredError(),
          'compound': models.losses.compound_loss_fss(loss_weights=LOSS_WEIGHTS),
          'compound_matteo': models.losses.compound_loss_fss_peak_matteo(LOSS_WEIGHTS),
        #   'compound_hess_weighted': models.losses.compound_loss_fss_hess_weighted(alpha=args.alpha),
        #   'compound_my_weighted': models.losses.compound_loss_fss_my_weighted(alpha=args.alpha,
        #                                                                       lambda_=args.lambda_,
        #                                                                       kappa=args.kappa),
          'hess_loss': models.losses.hess_loss(),
        #  'compound_cira_fss': models.losses.compound_cira_fss(loss_weights=LOSS_WEIGHTS),
          }
loss_func = losses[args.loss_name]

# Save params
#SAVE_NAME = f'tag.{args.name_tag}_loss.{args.loss_name}_n.21_peakweight.{args.peak_weight}_alpha.{args.alpha}_kappa{args.kappa}_blocks.{args.num_blocks}_mult.{args.mult_factor}'
SAVE_NAME = f'best_model.lr_{args.learning_rate}'
SAVE_PATH = f'results/{SAVE_NAME}'
exists = os.path.exists(SAVE_PATH)
if not exists:
    os.mkdir(SAVE_PATH)


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
            ids_batch = ids[start:end]
            for idx in ids_batch:
                era5, mswep = load_data(f'data/train_2001_over34/merged/merged_{idx}.nc')
                x_batch.append(era5)
                y_batch.append(mswep)
            # form batch and yield
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch

# load .csv with information on file names
df = pd.read_csv('data/train_2001_over34/ibtracs_2001_train_over34.csv').sample(frac=1)
ids = list(df.IDX_TRUE)

# split into 80% train 20% val
ids_train, ids_valid = train_test_split(ids, test_size=0.2, random_state=42)


"""
Callbacks
"""
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
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
if args.num_blocks == 3:
    model = models.models.arunet_threeblock(input_shape=INPUT_SHAPE,
                                            filters=UNET_FILTERS,
                                            loss_func=loss_func,
                                            initializer=initializer,
                                            #learning_rate=args.learning_rate,
                                            )
elif args.num_blocks == 4:
    model = models.models.arunet_fourblock(input_shape=INPUT_SHAPE,
                                           filters=UNET_FILTERS,
                                           loss_func=loss_func,
                                           initializer=initializer
                                           #learning_rate=args.learning_rate,
                                           )
#model.summary()
else:
    raise Exception("Number of UNet blocks specified is not currently implemented.")

# Train model
print(f'Training model: {SAVE_NAME}')
history = model.fit(train_generator(ids=ids_train),
                    steps_per_epoch=int(np.ceil(len(ids_train) / args.batch_size)),
                    epochs=args.epochs,
                    callbacks=callbacks,
                    validation_data=train_generator(ids_valid),
                    validation_steps=int(np.ceil(len(ids_valid) / args.batch_size)),
                    verbose=1)

print("FINISHED TRAINING; STARTING INFERENCE")


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

end_train = time.time()
training_time = (end_train - start)/60
print(f'Training took: {training_time:.2f} mins')

"""
Inference
"""
SAVE_FILES = False
MODE = 'test'
N = 15
ALPHA = 0.5

mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()
fss = models.losses.fss_loss_filtered

model_compound = model
df = pd.read_csv('data/test_2001_over34/ibtracs_2001_test_over34.csv')

# Data loading helper functions
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


# Metrics computation helper functions
def find_square(data, peak_row, peak_col):
    start_row = max(0, peak_row - N)
    end_row = min(96, peak_row + N + 1)
    start_col = max(0, peak_col - N)
    end_col = min(96, peak_col + N + 1)
    return data[start_row:end_row, start_col:end_col]

def compute_peak_distance(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_true_peak_loc = np.unravel_index(np.argmax(y_true), y_true.shape)
    y_pred_peak_loc = np.unravel_index(np.argmax(y_pred), y_pred.shape)
    err_x = (y_pred_peak_loc[0] - y_true_peak_loc[0]) * 11.1
    err_y = (y_pred_peak_loc[1] - y_true_peak_loc[1]) * 11.1
    dist = np.sqrt(err_x ** 2 + err_y**2)
    return (err_x, err_y, dist)

def compute_npeak_err(y_true, y_pred):
    peak_true = np.max(y_true)
    peak_loc_y_true = np.unravel_index(np.argmax(y_true, axis=None), y_true.shape)
    peak_row, peak_col = peak_loc_y_true[0], peak_loc_y_true[1]

    npeak_pred = np.max(find_square(y_pred, peak_row, peak_col))
    npeak_err = np.abs(peak_true - npeak_pred)

    return npeak_err

def compute_metrics(y_true, y_pred):
    metrics = {}

    # Pixel-level metrics
    metrics['mae'] = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred).numpy()
    metrics['mse'] = tf.keras.losses.MeanSquaredError()(y_true, y_pred).numpy()

    # FSS metrics
    metrics['fss_q80'] = fss(y_true, y_pred, n=15, q=80).numpy()
    metrics['fss_q95'] = fss(y_true, y_pred, n=15, q=95).numpy()
    metrics['fss_q99'] = fss(y_true, y_pred, n=15, q=99).numpy()

    # Image-level metrics
    corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    metrics['corr'] = corr
    metrics['percent_delta_bias'] = (np.sum(y_pred) - np.sum(y_true)) * 100 / np.sum(y_true)
    metrics['abs_delta_bias'] = np.abs(np.sum(y_true) - np.sum(y_pred))

    # Peak metrics
    metrics['peak_dist_x'], metrics['peak_dist_y'], metrics['peak_dist'] = compute_peak_distance(y_true, y_pred)
    metrics['npeak_err'] = compute_npeak_err(y_true, y_pred)

    return metrics


for idx, row in df.iterrows():
    # Load data
    IDX = df.loc[idx, 'IDX_TRUE']
    path = f'data/{MODE}_2001_over34/merged/merged_{IDX}.nc'

    x_batch, y_batch = load_data(path)
    x_batch = np.expand_dims(x_batch, axis=0)
    y_batch = np.expand_dims(y_batch, axis=0)

    # Make model prediction
    y_compound = model_compound.predict(x_batch, verbose=0)
    
    # Make sure to only select the 'total precipitation' field and to cut off the padding
    x_batch = x_batch[0, 2:-3, 2:-3, 0]
    y_batch = y_batch[0, 2:-3, 2:-3, 0]
    y_compound = y_compound[0, 2:-3, 2:-3, 0]

    # MSWEP
    df.loc[idx, 'MSWEP_peak'] = np.max(y_batch)
    df.loc[idx, 'MSWEP_tot'] = np.sum(y_batch)

    # ERA5
    df.loc[idx, 'ERA5_peak'] = np.max(x_batch)
    df.loc[idx, 'ERA5_tot'] = np.sum(x_batch)
    era5_metrics = compute_metrics(y_batch, x_batch)

    # compound
    df.loc[idx, f'Compound loss_peak'] = np.max(y_compound)
    df.loc[idx, f'Compound loss_tot'] = np.sum(y_compound)
    compound_metrics = compute_metrics(y_batch, y_compound)

    # Add metrics to the dataframe's row
    for metric_key in era5_metrics.keys():
        # Construct column names dynamically based on the metric key
        era5_col_name = f'ERA5_{metric_key}'
        compound_col_name = f'Compound loss_{metric_key}'

        # Assign the metric values to the corresponding columns for the current row
        df.loc[idx, era5_col_name] = era5_metrics[metric_key]
        df.loc[idx, compound_col_name] = compound_metrics[metric_key]


model_names = ['ERA5', 'Compound loss']
metrics_names = list(era5_metrics.keys())

# Create a new dataframe with the specified column names
df_means = pd.DataFrame(columns=['name'] + metrics_names)

# Fill in the 'name' column with the model names
df_means['name'] = model_names

# Now compute the mean for each metric and each model and store it in the new dataframe
for model_name in model_names:
    for metric_name in metrics_names:
        column_name = f"{model_name}_{metric_name}"  # Name of the column in df_all
        df_means.loc[df_means['name'] == model_name, metric_name] = np.mean(df[column_name])

# Convert the metric columns to numeric, as they are stored as objects after the above operation
df_means[metrics_names] = df_means[metrics_names].apply(pd.to_numeric)
ERA5_metrics = df_means.loc[df_means['name'] == 'ERA5'].values.tolist()[0][1:]
compound_metrics = df_means.loc[df_means['name'] == 'Compound loss'].values.tolist()[0][1:]


# Determine the maximum width for the name column and set fixed width for numbers
name_column_width = max(map(len, metrics_names))
number_column_width = 10  # Fixed width for numbers to account for decimal places and padding

# Header format string using the column widths
header_format = "{:<{name_width}}  {:>{num_width}}  {:>{num_width}}"
header_row = header_format.format("Metric Name", "ERA5", "Compound loss", name_width=name_column_width, num_width=number_column_width)

print("-" * len(header_row))  # Print a separator line
# Print the header
print(header_row)
print("-" * len(header_row))  # Print a separator line

# Row format string using fixed number width and 4 decimal places
row_format = "{:<{name_width}}  {:>{num_width}.4f}  {:>{num_width}.4f}"

# Print the rows
for name, era5, compound in zip(metrics_names, ERA5_metrics, compound_metrics):
    print(row_format.format(name, era5, compound, name_width=name_column_width, num_width=number_column_width))
print("-" * len(header_row))  # Print a separator line

end_inference = time.time()
inference_time = (end_inference - end_training)/60
print(f'Inference time: {inference_time:.2f} mins')

print(f'FINISHED! Model: {SAVE_NAME}')
