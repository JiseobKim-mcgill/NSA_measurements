"""
Created by : Jiseob Kim, McGill University (jiseob.kim@mcgill.ca)
Created    : [Dec 11, 2024], Last Update : [Dec 11, 2024]
Version    : 1.0

Description:
This script processes 'KAZR' and 'ceilometer' data by reading NetCDF files,
extracting key variables, and combining them for further analysis.

Usage: Run this script with `python script_name.py`.

This code is the property of the author.

Changelog:
- [Dec 11, 2024] Version 1.0: Initial version.
- [xxx xx, 2024] Version 1.x: ?
"""

import os
import netCDF4
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#import matplotlib.colors as mcolors
from datetime import datetime, timedelta

# ===================================================
# Utility Functions
# ===================================================
def nearest_time_indices(all_times, target_times, max_diff_seconds=30):
    """
    Given sorted arrays of all_times and target_times (both numpy arrays of datetime objects),
    find the nearest index in all_times for each target_time. If no time within max_diff_seconds is found,
    returns None for that index.

    Args:
        all_times (np.ndarray):    1D array of sorted datetime objects.
        target_times (np.ndarray): 1D array of sorted datetime objects to find closest in all_times.
        max_diff_seconds (int):    Maximum allowed time difference in seconds.

    Returns:
        np.ndarray: Array of indices into all_times that are closest to each target_time.
                    If no close time is found within max_diff_seconds, None is placed.
    """
    # Use searchsorted to find the insertion positions
    idxs = np.searchsorted(all_times, target_times)

    # For each target_time, the closest time in all_times could be at idxs[i] or idxs[i]-1.
    # We'll compare both (if valid).
    closest_indices = []
    for i, t in enumerate(target_times):
        pos = idxs[i]

        candidates = []
        if pos < len(all_times):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos-1)

        if not candidates:
            # no candidates
            closest_indices.append(None)
            continue

        # Find the candidate closest in time
        best_idx = None
        best_diff = None
        for c in candidates:
            diff_seconds = abs(((all_times[c] - t) / np.timedelta64(1, 's')).item())
            diff_seconds = int(round(diff_seconds))
            if best_diff is None or diff_seconds < best_diff:
                best_diff = diff_seconds
                best_idx = c

        # Check if within max_diff_seconds
        if best_diff <= max_diff_seconds:
            closest_indices.append(best_idx)
        else:
            closest_indices.append(None)

    return np.array(closest_indices, dtype=object)


def read_kazr_files(file_path, snr_threshold=-15, limit=None):
    """
    Read and process KAZR NetCDF files from the specified directory.

    Args:
        file_path (str): Directory containing KAZR NetCDF files.
        snr_threshold (float): Threshold for singal-to-noise ratio filtering
        limit (int, optional): Number of files to process. If None, processes all files.

    Returns:
        tuple: Combined time, reflectivity, and range data.
    """
    # Get the list of NetCDF in the directory
    kazr_files = sorted([os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.nc')])

    if limit is not None:
        kazr_files = kazr_files[:limit]
    
    print(f"Found {len(kazr_files)} KAZR files", flush=True)

    range_combined = None

    # Generate global target times (1-minute intervals)
    start_time = datetime(2023, 11, 1, 0, 0, 0)
    end_time = datetime(2024, 5, 1, 0, 0, 0)
    total_minutes = int((end_time - start_time).total_seconds() // 60)
    target_times = np.array([start_time + timedelta(minutes=i) for i in range(total_minutes)], dtype='datetime64[us]')

    # Combined data storage
    all_times = []
    all_reflectivity = []
    range_combined = None
    reflectivity_FillValue = None

    for kazr_file in kazr_files:
        with netCDF4.Dataset(kazr_file, mode='r') as nc_file:
            # Extract variables
            time_var = nc_file.variables['time']
            range_var = nc_file.variables['range']
            reflectivity_var = nc_file.variables['reflectivity']
            snr_copol_var = nc_file.variables['signal_to_noise_ratio_copolar_h']

            # Range [m]
            current_range = range_var[:]
            if range_combined is None:
                range_combined = current_range
            else:
                if not np.allclose(range_combined, current_range):
                    raise ValueError(f"Mismatch in range values detected in file: {kazr_file}")

            # Time [sec]
            base_time_str = time_var.units.split(' since ')[1]
            base_time_str = base_time_str.split(' ')[0] + ' ' + base_time_str.split(' ')[1]
            base_time = datetime.strptime(base_time_str, '%Y-%m-%d %H:%M:%S')
            time_datetime = np.array([base_time + timedelta(seconds=float(t)) for t in time_var[:]], dtype='datetime64[us]')

            # Reflectivity [dBZ]
            reflectivity = reflectivity_var[:]
            reflectivity_FillValue = reflectivity_var._FillValue
            reflectivity = np.where(snr_copol_var[:] < snr_threshold, reflectivity_FillValue, reflectivity)

            # Append current file data to global storage
            all_times.append(time_datetime)
            all_reflectivity.append(reflectivity)
        
    # Convert all times and reflectivity to numpy arrays
    all_times = np.concatenate(all_times)
    all_reflectivity = np.concatenate(all_reflectivity, axis=0)

    # Sort by time to ensure proper alignment
    sorted_indices = np.argsort(all_times)
    all_times = all_times[sorted_indices]
    all_reflectivity = all_reflectivity[sorted_indices]

    print(f"Extrating kazr data...", flush=True)
    closest_indices = nearest_time_indices(all_times, target_times, max_diff_seconds=30)
    # Build the final arrays
    time_combined = target_times.astype('O')  # convert back to datetime64 -> object datetime
    reflectivity_combined = []
    for i, idx in enumerate(closest_indices):
        if (i % (24*60)) == 0:
            print(f"- Processing {time_combined[i]}...", flush=True)
        if idx is None:
            reflectivity_combined.append(np.full_like(all_reflectivity[0, :], reflectivity_FillValue))
        else:
            reflectivity_combined.append(all_reflectivity[idx, :])

    reflectivity_combined = np.array(reflectivity_combined, dtype=np.float32)

    return time_combined, reflectivity_combined, range_combined, reflectivity_FillValue


def read_ceil_files(file_path, limit=None):
    """
    Read and process Ceilometer NetCDF files from the specified directory.

    Args:
        file_path (str): Path to the directory containing Ceilometer NetCDF file.
        limit (int, optional): Number of files to process. If None, processes all files.
    """
    # Get the list of NetCDF in the directory
    ceil_files = sorted([os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.nc')])

    if limit is not None:
        ceil_files = ceil_files[:limit]
    
    print(f"Found {len(ceil_files)} Ceilometer files", flush=True)

    # Generate global target times (1-minute intervals)
    start_time = datetime(2023, 11, 1, 0, 0, 0)
    end_time = datetime(2024, 5, 1, 0, 0, 0)
    total_minutes = int((end_time - start_time).total_seconds() // 60)
    target_times = np.array([start_time + timedelta(minutes=i) for i in range(total_minutes)], dtype='datetime64[us]')

    # Combined data storage
    all_times = []
    all_first_cbh = []
    all_second_cbh = []
    all_third_cbh = []
    missing_value = None

    for ceil_file in ceil_files:
        with netCDF4.Dataset(ceil_file, mode='r') as nc_file:
            # Extract variables
            time_var = nc_file.variables['time']
            first_cbh_var = nc_file.variables['first_cbh']
            second_cbh_var = nc_file.variables['second_cbh']
            third_cbh_var = nc_file.variables['third_cbh']

            # Time [sec]
            base_time_str = time_var.units.split(' since ')[1]
            base_time_str = base_time_str.split(' ')[0] + ' ' + base_time_str.split(' ')[1]
            base_time = datetime.strptime(base_time_str, '%Y-%m-%d %H:%M:%S')
            time_datetime = np.array([base_time + timedelta(seconds=float(t)) for t in time_var[:]], dtype='datetime64[us]')

            # Cloud base heights [m]
            missing_value = first_cbh_var.missing_value
            first_cbh = np.where(first_cbh_var[:] == missing_value, np.nan, first_cbh_var[:])
            second_cbh = np.where(second_cbh_var[:] == missing_value, np.nan, second_cbh_var[:])
            third_cbh = np.where(third_cbh_var[:] == missing_value, np.nan, third_cbh_var[:])

            # Append current file data to global storage
            all_times.append(time_datetime)
            all_first_cbh.append(first_cbh)
            all_second_cbh.append(second_cbh)
            all_third_cbh.append(third_cbh)

    # Convert all times and reflectivity to numpy arrays
    all_times = np.concatenate(all_times)
    all_first_cbh = np.concatenate(all_first_cbh)
    all_second_cbh = np.concatenate(all_second_cbh)
    all_third_cbh = np.concatenate(all_third_cbh)

    # Sort by time to ensure proper alignment
    sorted_indices = np.argsort(all_times)
    all_times = all_times[sorted_indices]
    all_first_cbh = all_first_cbh[sorted_indices]
    all_second_cbh = all_second_cbh[sorted_indices]
    all_third_cbh = all_third_cbh[sorted_indices]

    # Find nearest indices for all target_times
    print(f"Extrating Ceilometer data...", flush=True)
    closest_indices = nearest_time_indices(all_times, target_times, max_diff_seconds=30)

    time_combined = target_times.astype('O')
    first_cbh_combined = []
    second_cbh_combined = []
    third_cbh_combined = []
    for i, idx in enumerate(closest_indices):
        if (i % (24*60)) == 0:
            print(f"- Processing {time_combined[i]}...", flush=True)
        if idx is None:
            first_cbh_combined.append(missing_value)
            second_cbh_combined.append(missing_value)
            third_cbh_combined.append(missing_value)
        else:
            val1 = all_first_cbh[idx]
            val2 = all_second_cbh[idx]
            val3 = all_third_cbh[idx]
            # Replace NaN with missing_value
            val1 = missing_value if np.isnan(val1) else val1
            val2 = missing_value if np.isnan(val2) else val2
            val3 = missing_value if np.isnan(val3) else val3
            first_cbh_combined.append(val1)
            second_cbh_combined.append(val2)
            third_cbh_combined.append(val3)

    first_cbh_combined = np.array(first_cbh_combined, dtype=float)
    second_cbh_combined = np.array(second_cbh_combined, dtype=float)
    third_cbh_combined = np.array(third_cbh_combined, dtype=float)

    return time_combined, first_cbh_combined, second_cbh_combined, third_cbh_combined, missing_value


def write_to_netcdf(output_path, time, reflectivity, range_data, fill_value, first_cbh, second_cbh, third_cbh, missing_value, compression_level=4):
    """
    Write processed data into a NetCDF4 file with attributes

    Args:
        output_path (str):          Path to save the NetCDF file.
        time (np.ndarray):          Array of datetime objects.
        reflectivity (np.ndarray):  Reflectivity data.
        range_data (np.ndarray):    Range data.
        fill_value:                 Fill value for the variables
        first_cbh (np.ndarray):     First cloud base height [m]
        second_cbh (np.ndarray):    Second cloud base height [m]
        third_cbh (np.ndarray):     Third cloud base height [m]
    """
    with netCDF4.Dataset(output_path, mode='w', format='NETCDF4') as nc_file:
        # Define dimensions
        nc_file.createDimension('time', len(time))
        nc_file.createDimension('range', len(range_data))

        # Define variables
        time_var = nc_file.createVariable('time', 'f8', ('time',), zlib=True, complevel=compression_level)
        range_var = nc_file.createVariable('range', 'f4', ('range'), zlib=True, complevel=compression_level)
        reflectivity_var = nc_file.createVariable('reflectivity', 'f4', ('time', 'range'), fill_value=fill_value, zlib=True, complevel=compression_level)
        first_cbh_var = nc_file.createVariable('first_cbh', 'f4', ('time'), fill_value=missing_value, zlib=True, complevel=compression_level)
        second_cbh_var = nc_file.createVariable('second_cbh', 'f4', ('time'), fill_value=missing_value, zlib=True, complevel=compression_level)
        third_cbh_var = nc_file.createVariable('third_cbh', 'f4', ('time'), fill_value=missing_value, zlib=True, complevel=compression_level)

        # Add attributes
        time_var.units = 'seconds since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        range_var.units = 'm'
        reflectivity_var.long_name = 'Equivalent reflectivity factor'
        reflectivity_var.units = 'dBZ'
        reflectivity_var.coordinates = 'elevation azimuth range'
        reflectivity_var.standard_name = 'equivalent_reflectivity_factor'
        reflectivity_var.description = 'Filtered reflectivity where SNR < -15 dB'
        first_cbh_var.long_name = 'Lowest cloud base height detected'
        first_cbh_var.units = 'm'
        first_cbh_var.valid_min = 0.0
        first_cbh_var.valid_max = 7700.0
        first_cbh_var.missing_value = missing_value
        second_cbh_var.long_name = 'Second cloud base height detected'
        second_cbh_var.units = 'm'
        second_cbh_var.valid_min = 0.0
        second_cbh_var.valid_max = 7700.0
        second_cbh_var.missing_value = missing_value
        third_cbh_var.long_name = 'Third cloud base height detected'
        third_cbh_var.units = 'm'
        third_cbh_var.valid_min = 0.0
        third_cbh_var.valid_max = 7700.0
        third_cbh_var.missing_value = missing_value

        # Assign data
        time_var[:] = netCDF4.date2num(time, units=time_var.units, calendar=time_var.calendar)
        range_var[:] = range_data
        reflectivity_var[:] = reflectivity
        first_cbh_var[:] = first_cbh
        second_cbh_var[:] = second_cbh
        third_cbh_var[:] = third_cbh

'''
def plot_kazr_reflectivity(time_kazr, range_kazr, reflectivity_kazr, first_cbh, second_cbh, third_cbh):
    """
    Plot the KAZR reflectivity data.

    Args:
        time_kazr (np.ndarray):         Time [sec]
        range_kazr (np.ndarray):        Range [m]
        reflectivity_kazr (np.ndarray): Reflectivity [dBZ]
        first_cbh (np.ndarray):         First cloud base height [m]
        second_cbh (np.ndarray):        Second cloud base height [m]
        third_cbh (np.ndarray):         Third cloud base height [m]
    """
    print("\nCreating reflectivity plot...")

    range_kazr = range_kazr / 1000.0
    time_kazr_mesh, range_kazr_mesh = np.meshgrid(time_kazr, range_kazr, indexing='ij')

    bounds = np.arange(-40, 12, 2)
    colors = ['#ffffff', '#e6e9f5', '#4153ba', '#314098', '#1d4f96', 
            '#17826e', '#08a863', '#28b459', '#47b750', '#62bc4e',
            '#7bc14c', '#a5ce46', '#f5ed4d', '#fae741', '#fdd73d',
            '#ffc83c', '#fdb63a', '#faa537', '#f79335', '#f68833',
            '#f47831', '#f26a26', '#e35a28', '#d04428', '#ad2121'] # 25 colors
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 5) ,constrained_layout=True)
    pcm = ax.pcolormesh(mdates.date2num(time_kazr_mesh), range_kazr_mesh, reflectivity_kazr,
                        cmap=cmap, norm=norm, shading='nearest', zorder=2) 
    
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.01, boundaries=bounds, ticks=bounds)
    cbar.set_label('Reflectivity [dBZ]', fontsize=14)
    cbar.set_ticks([-40, -30, -20, -10, 0, 10])

    ax.set_xlabel('Time (UTC)', fontsize=16)
    ax.set_ylabel('Height [km]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    locator = mdates.HourLocator(interval=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))

    ax.set_ylim([-0.2, 10])

    # Add black dots for first_cbh where it is not missing_value
    missing_value = -9999.0  # Define the missing value
    valid_indices = first_cbh != missing_value  # Identify valid (non-missing) indices
    ax.scatter(mdates.date2num(time_kazr[valid_indices]), first_cbh[valid_indices] / 1000.0, 
                color='black', s=3, zorder=3, label='First CBH')  # Black dots
    valid_indices = second_cbh != missing_value  # Identify valid (non-missing) indices
    ax.scatter(mdates.date2num(time_kazr[valid_indices]), second_cbh[valid_indices] / 1000.0, 
                color='gray', s=3, zorder=3, label='Second CBH')  # Gray dots
    valid_indices = third_cbh != missing_value  # Identify valid (non-missing) indices
    ax.scatter(mdates.date2num(time_kazr[valid_indices]), third_cbh[valid_indices] / 1000.0, 
                color='lightgray', s=3, zorder=3, label='Third CBH')  # Light gray dots

    plt.savefig(f"/home/jskim/projects/ctb-itan/jskim/data/intermediate/Arctic_clouds/Reflectivity_KAZR_test.png", dpi=600)
    plt.close()
'''

# ===================================================
# Main Function
# ===================================================
def main():
    """
    Main function to orchestrate the processing of KAZR and ceilometer data and writing to NetCDF.
    """
    kazr_file_path = os.path.expanduser('/home/jskim/projects/ctb-itan/jskim/data/raw/NSA/LLC_for_WIVERN/kazrcfrge')
    ceil_file_path = os.path.expanduser('/home/jskim/projects/ctb-itan/jskim/data/raw/NSA/LLC_for_WIVERN/ceil')
    output_file_path = os.path.expanduser('/home/jskim/projects/ctb-itan/jskim/data/intermediate/WIVERN/kazr_and_ceil_20231101-20240430.nc')

    # Read and process KAZR files
    print("\nStarting KAZR data processing...", flush=True)
    time_kazr, reflectivity_kazr, range_kazr, fill_value = read_kazr_files(kazr_file_path, snr_threshold=-15, limit=None)

    print("\nProcessed KAZR data:", flush=True)
    print(f"- Total time points: {time_kazr.shape}", flush=True)
    print(f"- Range points: {range_kazr.shape}", flush=True)
    print(f"- Reflectivity shape: {reflectivity_kazr.shape}", flush=True)

    # Read and process Ceilometer files
    print("\nStarting Ceilometer data processing...", flush=True)
    time_ceil, first_cbh, second_cbh, third_cbh, missing_value = read_ceil_files(ceil_file_path, limit=None)

    print("\nProcessed Ceil data:", flush=True)
    print(f"- Total time points: {time_ceil.shape}", flush=True)
    print(f"- CBH shape: {first_cbh.shape}", flush=True)

    #plot_kazr_reflectivity(time_kazr, range_kazr, reflectivity_kazr, first_cbh, second_cbh, third_cbh)

    # Write processed data to NetCDF
    print(f"\nWriting {output_file_path}...", flush=True)
    write_to_netcdf(output_file_path, time_kazr, reflectivity_kazr, range_kazr, fill_value, first_cbh, second_cbh, third_cbh, missing_value) 
    print(f"Data written to {output_file_path}", flush=True)

    print("\n[DONE] Processed and saved KAZR data.", flush=True)


# ===================================================
# Main
# ===================================================
if __name__ == "__main__":
    main()