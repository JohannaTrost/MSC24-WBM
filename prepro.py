import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime
from tqdm import tqdm


def split_dataset(data, save=False, path=''):

    # Convert to date object first
    data['date'] = pd.to_datetime(data['date'])

    # Split time series
    df_2000_to_2011 = data[(data['date'].dt.year >= 2000) & (
            data['date'].dt.year <= 2011)]
    df_2012_to_2023 = data[(data['date'].dt.year >= 2012) & (
            data['date'].dt.year <= 2023)]

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


def load_precip_ncs(save=False):
    """Load nc file for all years (2000-2023) and merge them into one pandas dataframe"""
    years = np.arange(2000, 2023, 1)

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
    df_2000_to_2011, df_2012_to_2023 = split_dataset(precip_df,
                                                     save=save,
                                                     path='data/'
                                                          'total_precipitation')

    return precip_df
