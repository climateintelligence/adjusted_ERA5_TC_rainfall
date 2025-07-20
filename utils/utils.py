import numpy as np
import xarray as xr
import h5py
import tensorflow as tf
from datetime import date, datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error


def normalise_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def convert_iso_time_3h(iso_time, return_id=False):
    """Converts a string containing an ISO time into an id along the time axis, with the
    convention that id = 0 is 1 January 1979 at 00:00, and each timestep is 3h.

    Arguments:
        iso_time [string]: The ISO time string to be converted.
        return_id [bool]: Whether to return the id that corresponds to that iso
                          time in an array that has 3h temporal resolution, or
                          to return the year, month, day, and hour that correspond
                          to the ISO time.

    Returns:
        [int]: The corresponding id.
    """
    day = iso_time.split('/')[0]
    month = iso_time.split('/')[1]
    year = iso_time.split('/')[2][:4]
    hour = iso_time[-5:-3]
    if return_id:
        day = int(day)
        month = int(month)
        year = int(year)
        hour = int(hour)
        start_date = date(1979, 1, 1)
        end_date = date(year, month, day)
        elapsed_days = end_date - start_date
        time_id = int(elapsed_days.days * 8 + hour/3)
        return time_id
    else:
        return year, month, day, hour

def add_to_iso_time(iso_time, hours_delta=0, minutes_delta=0):
    """Adds 3*steps hours to the given iso_time.

    Arguments:
        iso_time [string]: The ISO time string to be extended.
        steps [int]: The number of steps to extend the iso_time by (each step is 3h).

    Returns:
        [int]: The corresponding id.
    """
    day = int(iso_time.split('/')[0])
    month = int(iso_time.split('/')[1])
    year = int(iso_time.split('/')[2][:4])
    hours = int(iso_time[-5:-3])
    minutes = int(iso_time[-2:])
    date = datetime(year, month, day, hours, minutes)
    extended_date = date + timedelta(hours=hours_delta, minutes=minutes_delta)

    extended_iso_time = f'{extended_date.day:02d}/{extended_date.month:02d}/{extended_date.year} {extended_date.hour:02d}:{extended_date.minute:02d}'
    #extended_iso_time = extended_iso_time.strftime('%Y-%m-%d %H:%M:%S')
    return extended_iso_time



def convert_box_radius(radius, unit='km', res=0.25):
    """Determines the length of the slice of array that will be used to define
       a box around the TC centre.

    Arguments:
        radius [int]: The desired box radius.
        unit [str]: The unit in which the radius is expressed ('deg' or 'km').
        res [float]: The resolution of the array that will be sliced.

    Returns:
        [int]: The corresponding id.
    """
    if unit=='km':
        radius = radius / 111 # CHECK THIS: should it be different at different points on the grid?
    elif unit != 'deg':
        raise ValueError('Unrecognised unit. Allowed units are: "deg", "km". Default unit is: "km".')
    box_radius = int(radius / res)
    return box_radius


def find_id(value, domain):
    return int(((value - domain)**2).argmin())


def box_var(var, lat_id, lon_id, box_radius, domain_lats, domain_lons, idx):
    """Outputs a slice of an input array (var) equal to a box of radius box_radius
       centred around a point with coordinates (lat_id, lon_id) inside of var.

    Arguments:
        var [np.ndarray]: The variable to be sliced.
        lat_id [int]: The lat coordinate of the centre of the box.
        lon_id [int]: The lon coordinate of the centre of the box.
        box_radius [int]: The radius of the box (in number of elements of var to
                          keep along each dimension).

    Returns:
        ra_TC_lat [int]: The latitude of the TC centre.
        ra_TC_lon [float]: The longitude of the TC centre.
        ra_TC_lat_id [int]: The index inside lats_domain corresponding to the TC centre.
        ra_TC_lon_id [int]: The index inside lons_domain corresponding to the TC centre.
        box_lats [np.ndarray]: The latitudes inside a 500 km box centred on the TC.
        box_lons [np.ndarray]: The longitudes inside a 500 km box centred on the TC.
    """
    left_margin_needed = lon_id - box_radius
    right_margin_needed = lon_id + box_radius+1 - domain_lons.shape[0]
    top_margin_needed = lat_id + box_radius+1
    bottom_margin_needed = lat_id - box_radius
    if left_margin_needed < 0:
        sliced = var[:,left_margin_needed:]
        var = xr.concat([sliced, var], dim='lon')
        lon_id -= left_margin_needed # it's a minus because left_margin_needed is negative
    elif right_margin_needed > 0:
        sliced = var[:, :right_margin_needed]
        var = xr.concat([var, sliced], dim='lon')
    var = var[lat_id-box_radius:lat_id+box_radius+1, lon_id-box_radius:lon_id+box_radius+1]
    return var


def uniform_lons_to_mswep(ds):
    """Uniform input dataset to MSWEP's longitude grid, which is [-180, 180].

       Arguments:
           ds [xr.Dataset]: The dataset whose longitude needs to be shifted.

       Returns:
           ds [xr.Dataset]: The input dataset with longitude dimension shifted.
    """
    _lon_names = ['lon', 'longitude']
    ds_dims = list(ds.dims)
    lon_name = list(set(_lon_names).intersection(ds_dims))[0]
    # Adjust lon values to make sure they are within (-180, 180)
    ds['lon_adjusted'] = xr.where(
        ds[lon_name] > 180,
        ds[lon_name] - 360,
        ds[lon_name])
    ds = (
        ds
        .swap_dims({lon_name: 'lon_adjusted'})
        .sel(**{'lon_adjusted': sorted(ds.lon_adjusted)})
        .drop(lon_name))

    ds = ds.rename({'lon_adjusted': lon_name})
    return ds


def interp_era5_to_mswep(era5_ds, mswep_ds):
    """Interpolate input era5 dataset to target mswep dataset (needed because
       the two are defined on different lat/lon grids).

       Arguments:
           era5_ds [xr.Dataset]: The ERA5 dataset that needs to be interpolated.
           mswep_ds [xr.Dataset]: The MSWEP dataset that has the desired lat/lon grid
                                  to which we are interpolating the ERA5 dataset.

       Returns:
           era5_interp_mswep [xr.Dataset]: The input ERA5 dataset, interpolated
                                           onto the input MSWEP dataset's grid.
    """
    era5_new_lons = uniform_lons_to_mswep(era5_ds)
    era5_interp_mswep = era5_new_lons.interp(longitude=mswep_ds.lon.values,
                                             latitude=mswep_ds.lat.values)
    return era5_interp_mswep


def slice_ds_timestep(ds, iso_time, varname):
    year, month, day, hour = convert_iso_time_3h(iso_time)
    current_time = np.datetime64(f'{year}-{month}-{day}T{hour}:00:00.000000000')
    timestep_idx = np.where(ds.time.values == current_time)[0][0]
    timestep_slice = ds[varname].isel(time=timestep_idx)
    return timestep_slice

# Create a function to load the data from a single file
def load_data(filename):
    decoded_filename = filename.decode()
    with xr.open_dataset(decoded_filename)['tp'] as era5_ds:
        era5_tp = np.array(era5_ds) * 1000
        era5_tp = np.pad(era5_tp, (2,3), constant_values=0)
        era5_tp = np.expand_dims(era5_tp,axis=-1)
    mswep_filename = decoded_filename.replace('ERA5', 'MSWEP')
    with xr.open_dataset(mswep_filename)['precipitation'] as mswep_ds:
        mswep_tp = np.array(mswep_ds)
        mswep_tp = np.pad(mswep_tp, (2,3), constant_values=0)
        mswep_tp = np.expand_dims(mswep_tp,axis=-1)
    return era5_tp, mswep_tp

# Create a function to load the data and preprocess it for training
def preprocess(filename):
    era5, mswep = tf.numpy_function(load_data, [filename], [tf.float32, tf.float32])
    # Add any additional preprocessing steps here
    return era5, mswep


# Create a function to load the data from a single file
def load_data_merged(filename):
    decoded_filename = filename.decode()
    with xr.open_dataset(decoded_filename) as ds:
        # read era5 tp data
        era5_tp = np.array(ds['tp'])
        era5_tp = np.pad(era5_tp, (2,3), constant_values=0)
        era5_tp = np.expand_dims(era5_tp,axis=-1)
        # read mswep tp data
        mswep_tp = np.array(ds['precipitation'])
        mswep_tp = np.pad(mswep_tp, (2,3), constant_values=0)
        mswep_tp = np.expand_dims(mswep_tp,axis=-1)
    return era5_tp, mswep_tp

# Create a function to load the data and preprocess it for training
def preprocess_merged(filename):
    era5, mswep = tf.numpy_function(load_data_merged, [filename], [tf.float32, tf.float32])
    return era5, mswep



def calculate_metrics(y_true, y_pred):
    # total precipitation metrics
    delta_bias = np.mean(y_true - y_pred)
    abs_delta_bias = np.mean(np.abs(y_true - y_pred))
    std = np.std(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return delta_bias, abs_delta_bias, std, mae