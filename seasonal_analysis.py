import os
import pandas as pd
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

out_half = {}
out_dbl = {}
out_normal = {}

for out in ['evapotranspiration', 'runoff', 'snow', 'soil_moisture']:
    out_half[out] = xr.open_dataset(f"results/half_precip_output/{out}.nc")
    out_normal[out] = xr.open_dataset(f"results/fast_model/{out}.nc")
    out_dbl[out] = xr.open_dataset(f"results/dbl_precip_output/{out}.nc")

out = 'evapotranspiration'

# Define the seasons
seasons = {
    'DJF': [12, 1, 2],  # December, January, February
    'MAM': [3, 4, 5],  # March, April, May
    'JJA': [6, 7, 8],  # June, July, August
    'SON': [9, 10, 11]  # September, October, November
}

# Group the data by season and calculate the mean
mean_data_normal_seasonal = out_normal[out].groupby('time.season').mean('time')
mean_data_scaled_seasonal = out_half[out].groupby('time.season').mean('time')

# Calculate the percent change for each season
perc_change_seasonal = {}
for season in seasons:
    # Select the data for the current season
    mean_data_season = mean_data_normal_seasonal.sel(season=season)[out]
    mean_data_scaled_season = mean_data_scaled_seasonal.sel(season=season)[out]

    # Calculate percent change
    perc_change_seasonal[season] = ((mean_data_scaled_season - mean_data_season) /
                                    mean_data_season) * 100

# Create a plot with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8),
                        subplot_kw={'projection': ccrs.PlateCarree()})

# Define seasons and corresponding subplot positions
seasons_months = ['DJF', 'MAM', 'JJA', 'SON']
seasons_names = {'DJF': 'Winter',
                 'MAM': 'Spring',
                 'JJA': 'Summer',
                 'SON': 'Fall'}
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Loop over seasons and subplot positions
for season, pos in zip(seasons_months, positions):
    # Select data for the current season
    data = perc_change_seasonal[season]

    # Get the corresponding subplot axis
    ax = axs[pos]

    # Add country borders and coastlines
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.coastlines()

    # Plot the data
    img = data.plot.imshow(x='lon', y='lat', cmap='viridis',
                           transform=ccrs.PlateCarree(),
                           ax=ax, add_colorbar=False)

    # Set title
    ax.set_title(seasons_names[season])

# Add colorbar
fig.colorbar(img, ax=axs, orientation='vertical', shrink=0.7,
             label='Percent Change')

plt.suptitle(f'Simulated mean {out} percent change by season')
plt.tight_layout()
plt.show()