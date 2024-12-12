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
from datetime import datetime, timedelta

# ===================================================
# Utility Functions
# ===================================================
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

    time_combined = []
    reflectivity_combined = []
    range_combined = None

    # Generate global target times (1-minute intervals)
    start_time = datetime(2023, 11, 1, 0, 0, 0)
    end_time = datetime(2024, 5, 1, 0, 0, 0)
    target_times = [start_time + timedelta(minutes=i) for i in range(int((end_time - start_time).total_seconds() // 60))]

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
            time_datetime = np.array([base_time + timedelta(seconds=t) for t in time_var[:]])

            # Reflectivity [dBZ]
            reflectivity = reflectivity_var[:]
            reflectivity_FillValue = reflectivity_var._FillValue
            reflectivity = np.where(snr_copol_var[:] < snr_threshold, reflectivity_FillValue, reflectivity)

            # Append current file data to global storage
            all_times.extend(time_datetime)
            all_reflectivity.extend(reflectivity)
        
    # Convert all times and reflectivity to numpy arrays
    all_times = np.array(all_times)
    all_reflectivity = np.array(all_reflectivity)

    # Sort by time to ensure proper alignment
    sorted_indices = np.argsort(all_times)
    all_times = all_times[sorted_indices]
    all_reflectivity = all_reflectivity[sorted_indices]

    # Match global target times
    time_combined = []
    reflectivity_combined = []

    print(f"Extrating kazr data...", flush=True)
    for target_time in target_times:
        if target_time.hour == 0 and target_time.minute == 0:
            print(f"- Processing {target_time}...", flush=True)
        time_diffs = np.abs((all_times - target_time).astype('timedelta64[s]').astype(int))
        closest_idx = np.argmin(time_diffs)
        closest_time = all_times[closest_idx]

        if abs((closest_time - target_time).total_seconds()) <= 30:
            time_combined.append(target_time)
            reflectivity_combined.append(all_reflectivity[closest_idx, :])
        else:
            time_combined.append(target_time)
            reflectivity_combined.append(np.full_like(all_reflectivity[0, :], reflectivity_FillValue))

    time_combined = np.array(time_combined)
    reflectivity_combined = np.array(reflectivity_combined)

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
    target_times = [start_time + timedelta(minutes=i) for i in range(int((end_time - start_time).total_seconds() // 60))]

    # Combined data storage
    all_times = []
    all_first_cbh = []
    all_second_cbh = []
    all_third_cbh = []

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
            time_datetime = np.array([base_time + timedelta(seconds=t) for t in time_var[:]])

            # Cloud base heights [m]
            missing_value = first_cbh_var.missing_value
            first_cbh = np.where(first_cbh_var[:] == missing_value, np.nan, first_cbh_var[:])
            second_cbh = np.where(second_cbh_var[:] == missing_value, np.nan, second_cbh_var[:])
            third_cbh = np.where(third_cbh_var[:] == missing_value, np.nan, third_cbh_var[:])

            # Append current file data to global storage
            all_times.extend(time_datetime)
            all_first_cbh.extend(first_cbh)
            all_second_cbh.extend(second_cbh)
            all_third_cbh.extend(third_cbh)

    # Convert all times and reflectivity to numpy arrays
    all_times = np.array(all_times)
    all_first_cbh = np.array(all_first_cbh)
    all_second_cbh = np.array(all_second_cbh)
    all_third_cbh = np.array(all_third_cbh)

    # Sort by time to ensure proper alignment
    sorted_indices = np.argsort(all_times)
    all_times = all_times[sorted_indices]
    all_first_cbh = all_first_cbh[sorted_indices]
    all_second_cbh = all_second_cbh[sorted_indices]
    all_third_cbh = all_third_cbh[sorted_indices]

    # Match global target times
    time_combined = []
    first_cbh_combined = []
    second_cbh_combined = []
    third_cbh_combined = []

    print(f"Extrating Ceilometer data...", flush=True)
    for target_time in target_times:
        if target_time.hour == 0 and target_time.minute == 0:
            print(f"- Processing {target_time}...", flush=True)
        time_diffs = np.abs((all_times - target_time).astype('timedelta64[s]').astype(int))
        closest_idx = np.argmin(time_diffs)
        closest_time = all_times[closest_idx]

        if abs((closest_time - target_time).total_seconds()) <= 30:
            time_combined.append(target_time)
            first_cbh_combined.append(all_first_cbh[closest_idx])
            second_cbh_combined.append(all_second_cbh[closest_idx])
            third_cbh_combined.append(all_third_cbh[closest_idx])
        else:
            time_combined.append(target_time)
            first_cbh_combined.append(np.full_like(all_first_cbh[0], missing_value))
            second_cbh_combined.append(np.full_like(all_second_cbh[0], missing_value))
            third_cbh_combined.append(np.full_like(all_third_cbh[0], missing_value))

    time_combined = np.array(time_combined)
    first_cbh_combined = np.array(first_cbh_combined)
    second_cbh_combined = np.array(second_cbh_combined)
    third_cbh_combined = np.array(third_cbh_combined)

    # Replace NaN values with missing_value
    first_cbh_combined = np.nan_to_num(first_cbh_combined, nan=missing_value)
    second_cbh_combined = np.nan_to_num(second_cbh_combined, nan=missing_value)
    third_cbh_combined = np.nan_to_num(third_cbh_combined, nan=missing_value)

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

# ===================================================
# Main Function
# ===================================================
def main():
    """
    Main function to orchestrate the processing of KAZR and ceilometer data and writing to NetCDF.
    """
    kazr_file_path = os.path.expanduser('/kazr/path')
    ceil_file_path = os.path.expanduser('/ceil/path')
    output_file_path = os.path.expanduser('/output/path/kazr_and_ceil_20231101-20240430.nc')

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
