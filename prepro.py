import os
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
from tqdm import tqdm
import xarray as xr


def load_data(start_year=2000, end_year=2024):
    # Read in forcing data
    years = np.arange(start_year, end_year, 1)
    P_data = []
    R_data = []
    T_data = []
    lai_data = []

    for year in years:
        file_path = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.' + str(
            year) + '.nc'
        P_data.append(xr.open_dataset(file_path)['tp'])

        file_path = 'data/net_radiation/nr.daily.calc.era5.0d50_CentralEurope.' + str(
            year) + '.nc'
        R_data.append(xr.open_dataset(file_path)['nr'])

        file_path = 'data/daily_average_temperature/t2m_mean.daily.calc.era5.0d50_CentralEurope.' + str(
            year) + '.nc'
        T_data.append(xr.open_dataset(file_path)['t2m'])

        file_path = 'data/lai/lai.daily.0d50_CentralEurope.' + str(year) + '.nc'
        lai_data.append(xr.open_dataset(file_path)['lai'])

    # Convert lists to xarray datasets
    P_data = xr.concat(P_data, dim='time')
    R_data = xr.concat(R_data, dim='time')
    T_data = xr.concat(T_data, dim='time')
    lai_data = xr.concat(lai_data, dim='time')

    return P_data, R_data, T_data, lai_data


def fill_missing_lai(lai_data):
    gridcells = []
    for lat in range(len(lai_data.latitude)):
        for lon in range(len(lai_data.longitude)):
            if np.isnan(lai_data[:, lat, lon]).all():
                gridcells.append((lat, lon))

    for lat in range(len(lai_data.latitude)):
        for lon in range(len(lai_data.longitude)):
            if np.isnan(lai_data[:, lat, lon]).all():
                continue
            else:
                # Get the data for the current grid cell
                grid_cell_data = lai_data[:, lat, lon]

                # Convert to DataFrame for easier handling
                grid_cell_df = pd.DataFrame(grid_cell_data)

                # Fill NA values with the previous value
                grid_cell_df.fillna(method='ffill', inplace=True)

                # Update the grid cell data
                lai_data[:, lat, lon] = grid_cell_df.values.flatten()

    return lai_data


def adjust_precipitation(precip, scale_factor=2, cutoff=1e-05):
    # Step 1: Scale the data to double its variability
    scaled_precip = precip * scale_factor

    # Step 2: Adjust for negativity
    adjusted_precip = np.where(scaled_precip < 0, 0, scaled_precip)
    adjusted_mean = adjusted_precip.mean()

    # Step 3: Iterative Offset Adjustment
    while np.abs(adjusted_mean - precip.mean()) > cutoff:
        offset = precip.mean() - adjusted_mean
        adjusted_precip += offset
        adjusted_precip = np.where(adjusted_precip < 0, 0, adjusted_precip)
        adjusted_mean = adjusted_precip.mean()

    # Step 4: Final Adjustment
    if (adjusted_precip < 0).any():
        min_value = adjusted_precip.min()
        adjusted_precip -= min_value

    return adjusted_precip


def split_dataset(data, save=False, path=''):

    # Convert to date object first
    data['date'] = pd.to_datetime(data['date'])

    # Split time series
    df_2000_to_2011 = data[(data['date'].dt.year >= 2000) & (
            data['date'].dt.year <= 2011)]
    df_2012_to_2023 = data[(data['date'].dt.year >= 2012)]

    if save:
        # Print the first few rows of each DataFrame to verify
        print("DataFrame from 2000 to 2011:")
        print(df_2000_to_2011.head())

        print("\nDataFrame from 2012 to 2023:")
        print(df_2012_to_2023.head())

        df_2000_to_2011.to_csv(f'{path}/precip_2000_2011.csv',
                               index=False)
        df_2012_to_2023.to_csv(f'{path}/precip_2012_2023.csv',
                               index=False)

    return df_2000_to_2011, df_2012_to_2023


def load_precip_ncs(save=False, start_year=2000, end_year=2024):
    """Load nc file for all years (2000-2023) and merge them into one pandas
    dataframe """
    years = np.arange(start_year, end_year)

    # Flatten the data
    data = []

    for year in tqdm(years):

        file_path = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.' + str(
            year) + '.nc'

        nc_file = nc.Dataset(file_path)
        lon = nc_file.variables['lon'][:]
        lat = nc_file.variables['lat'][:]
        dates = nc_file.variables['time'][:]

        for i, date in enumerate(dates):
            for j in range(len(lat)):
                for k in range(len(lon)):
                    tp = nc_file.variables['tp'][i, j, k]
                    # Convert day of the year to a timestamp
                    year_start = datetime(year, 1, 1)
                    timestamp = year_start + pd.Timedelta(days=int(date))
                    data.append([timestamp, lat[j], lon[k], tp])

        nc_file.close()

    # Create DataFrame
    precip_df = pd.DataFrame(data, columns=['date', 'lat', 'lon', 'tp'])

    precip_df['tp'] = np.round([d_year[-1].item() for d_year in data], 10)

    print(precip_df.head())

    # Split df for saving 
    split_dataset(precip_df, save=save, path='data/total_precipitation')

    return precip_df
