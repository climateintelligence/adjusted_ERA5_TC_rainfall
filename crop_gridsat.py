import xarray as xr
import numpy as np
import glob
from datetime import datetime, timedelta
import scipy.ndimage 

MODE = 'train'
ERA5_PATH = f'data/{MODE}_2001_over34/merged/'
GRIDSAT_PATH = f'data/{MODE}_2001_over34/gridsat/'
FINAL_SAVE_PATH = f'data/{MODE}_2001_over34/merged_gridsat/'

def decompose_datetime(dt):
    year = f'{dt.year}'
    month = f'{dt.month:02d}'
    day = f'{dt.day:02d}'
    time = f'{dt.hour:02d}'
    return year, month, day, time

def downsample(original_array, target_dims=[91, 91]):
#     if original_height != original_width:
        
    # Desired dimensions
    target_height, target_width = target_dims
    # Calculating the zoom factors for each dimension
    zoom_factor_height = target_height / original_array.shape[0]
    zoom_factor_width = target_width / original_array.shape[1]
    # Downsampling using zoom
    downsampled_array = scipy.ndimage.zoom(original_array, (zoom_factor_height, zoom_factor_width))
    return downsampled_array

era5_filenames = glob.glob(f'data/{MODE}_2001_over34/merged/*.nc')


for filename in era5_filenames:
    with xr.open_dataset(filename) as ds:
        idx_true = filename[-9:-3]
        save_name = f'{FINAL_SAVE_PATH}gridsat_{idx_true}.nc'
        era5 = ds.tp
        lons = era5.longitude.values
        lats = era5.latitude.values
        box = [lats[0].astype(float),
               lons[0].astype(float),
               lats[-1].astype(float),
               lons[-1].astype(float)]
        first_time = datetime.utcfromtimestamp(ds.time.values.astype('O')/1e9)
        second_time = first_time + timedelta(hours=3)
        
        first_year, first_month, first_day, first_hour = decompose_datetime(first_time)
        second_year, second_month, second_day, second_hour = decompose_datetime(second_time)
        
        first_filename = GRIDSAT_PATH + f'GRIDSAT-B1.{first_year}.{first_month}.{first_day}.{first_hour}.nc'
        second_filename = GRIDSAT_PATH + f'GRIDSAT-B1.{second_year}.{second_month}.{second_day}.{second_hour}.nc'
        with xr.open_dataset(first_filename)['irwin_cdr'] as first_ds:
            cropped_first_ds = first_ds.sel(lat=slice(box[2],box[0]), lon=slice(box[1],box[3]))
            cropped_first_arr = np.array(cropped_first_ds[0, :, :])
        with xr.open_dataset(second_filename)['irwin_cdr'] as second_ds:
            cropped_second_ds = second_ds.sel(lat=slice(box[2],box[0]), lon=slice(box[1],box[3]))
            cropped_second_arr = np.array(cropped_second_ds[0, :, :])
        cropped_mean_arr = (cropped_first_arr + cropped_second_arr) / 2
        cropped_mean_down_arr = downsample(cropped_mean_arr, target_dims=ds['tp'].shape)
        ds['gridsat'] = (('latitude', 'longitude'), cropped_mean_down_arr)