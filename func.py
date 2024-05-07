import numpy as np
import xarray as xr
import run
from scipy.signal import detrend
from matplotlib import pyplot as plt
import rioxarray
import pyproj
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import pandas as pd

# Function to run model per gridcell
def grid_model(P_data, R_data, T_data, lai_data, params, cell = False):
    # Create empty arrays to store the output data
    snow = np.zeros_like(P_data)
    soil_moisture = np.zeros_like(P_data)
    evapotranspiration = np.zeros_like(P_data)
    runoff = np.zeros_like(P_data)
    for lat in range(len(P_data.lat)):
        for lon in range(len(P_data.lon)):

            if not cell:
                if lat != 0 or lon != 0:
                    break

            if np.isnan(lai_data[:, lat, lon]).all():
                print("Removed grid cell:", lat, lon)
                continue
            R_data_grid = R_data[:, lat, lon]
            P_data_grid = P_data[:, lat, lon]
            T_data_grid = T_data[:, lat, lon]
            lai_data_grid = lai_data[:, lat, lon]

            # Run Model for daily values
            daily_output = run.time_evolution(P_data_grid, R_data_grid, T_data_grid, lai_data_grid, *params)

            snow[:, lat, lon] = daily_output['snow'].values
            soil_moisture[:, lat, lon] = daily_output['calculated_soil_moisture'].values
            evapotranspiration[:, lat, lon] = daily_output['evapotranspiration'].values
            runoff[:, lat, lon] = daily_output['runoff'].values
    
    
    # Convert arrays to xarrays with corresponding latitudes and longitudes
    snow_xr = xr.DataArray(snow, dims=('time', 'lat', 'lon'), coords={'time': P_data.time, 'lat': P_data.lat, 'lon': P_data.lon})
    soil_moisture_xr = xr.DataArray(soil_moisture, dims=('time', 'lat', 'lon'), coords={'time': P_data.time, 'lat': P_data.lat, 'lon': P_data.lon})
    evapotranspiration_xr = xr.DataArray(evapotranspiration, dims=('time', 'lat', 'lon'), coords={'time': P_data.time, 'lat': P_data.lat, 'lon': P_data.lon})
    runoff_xr = xr.DataArray(runoff, dims=('time', 'lat', 'lon'), coords={'time': P_data.time, 'lat': P_data.lat, 'lon': P_data.lon})


    # Merge DataArrays into a Dataset
    output_dataset = xr.Dataset({
        'snow': snow_xr,
        'soil_moisture': soil_moisture_xr,
        'evapotranspiration': evapotranspiration_xr,
        'runoff': runoff_xr
    })
    
    return output_dataset


# Function to only calculate certain gridcells
def grid_model_cell(P_data, R_data, T_data, lai_data, params, cells):
    # Create empty arrays to store the output data
    snow = np.zeros_like(P_data)
    soil_moisture = np.zeros_like(P_data)
    evapotranspiration = np.zeros_like(P_data)
    runoff = np.zeros_like(P_data)
    for i in cells:
        if np.isnan(lai_data[:, i[0], i[1]]).all():
            print("Removed grid cell:", i[0], i[1])
            continue
        R_data_grid = R_data[:, i[0], i[1]]
        P_data_grid = P_data[:, i[0], i[1]]
        T_data_grid = T_data[:, i[0], i[1]]
        lai_data_grid = lai_data[:, i[0], i[1]]

        # Run Model for daily values
        daily_output = run.time_evolution(P_data_grid, R_data_grid, T_data_grid, lai_data_grid, *params)

        snow[:, i[0], i[1]] = daily_output['snow'].values
        soil_moisture[:, i[0], i[1]] = daily_output['calculated_soil_moisture'].values
        evapotranspiration[:, i[0], i[1]] = daily_output['evapotranspiration'].values
        runoff[:, i[0], i[1]] = daily_output['runoff'].values
        print(i, "done")
    
    
    # Convert arrays to xarrays with corresponding latitudes and longitudes
    snow_xr = xr.DataArray(snow, dims=('time', 'lat', 'lon'), coords={'time': P_data.time, 'lat': P_data.lat, 'lon': P_data.lon})
    soil_moisture_xr = xr.DataArray(soil_moisture, dims=('time', 'lat', 'lon'), coords={'time': P_data.time, 'lat': P_data.lat, 'lon': P_data.lon})
    evapotranspiration_xr = xr.DataArray(evapotranspiration, dims=('time', 'lat', 'lon'), coords={'time': P_data.time, 'lat': P_data.lat, 'lon': P_data.lon})
    runoff_xr = xr.DataArray(runoff, dims=('time', 'lat', 'lon'), coords={'time': P_data.time, 'lat': P_data.lat, 'lon': P_data.lon})


    # Merge DataArrays into a Dataset
    output_dataset = xr.Dataset({
        'snow': snow_xr,
        'soil_moisture': soil_moisture_xr,
        'evapotranspiration': evapotranspiration_xr,
        'runoff': runoff_xr
    })
    
    return output_dataset


# Remove linear trend without removing mean
def rem_trend(data):
    # Convert temperature data to numpy array
    data_array = data.values

    # Reshape the data into 2D array (time, flattened spatial dimensions)
    time_length = len(data.time)
    spatial_dims = data.shape[1:]
    data_2d = data_array.reshape((time_length, -1))

    # Detrend the data to remove the linear trend
    detrended_data = detrend(data_2d, axis=0, type='linear')
    
    # Calculate the mean of the original data
    mean_data = np.mean(data_2d, axis=0)
    
    # Add back the mean to the detrended data
    detrended_data_with_mean = detrended_data + mean_data

    # Reshape detrended data back to its original shape
    output = xr.DataArray(detrended_data_with_mean.reshape((time_length,) + spatial_dims),
                                            coords=data.coords, dims=data.dims)
    
    return output


# Plot function
def plot_func(data, grid,source):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plotting the first subplot
    data[source].isel(lat=grid[0][0], lon=grid[0][1]).plot(x='time', ax=axs[0, 0])
    axs[0, 0]

    # Plotting the second subplot
    data[source].isel(lat=grid[1][0], lon=grid[1][1]).plot(x='time', ax=axs[0, 1])
    axs[0, 1]

    # Plotting the third subplot
    data[source].isel(lat=grid[2][0], lon=grid[2][1]).plot(x='time', ax=axs[1, 0])
    axs[1, 0]

    # Plotting the fourth subplot
    data[source].isel(lat=grid[3][0], lon=grid[3][1]).plot(x='time', ax=axs[1, 1])
    axs[1, 1]

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()



# function to clip and interpolation projections data
def clip_proj(forcing_data, proj_data):
    
    min_lon = forcing_data.lon.min().values - 0.5
    min_lat = forcing_data.lat.min().values - 0.5
    max_lon = forcing_data.lon.max().values + 0.5
    max_lat = forcing_data.lat.max().values + 0.5
    
    proj_data = proj_data.rio.write_crs('epsg:4326')

    # Clip the projection data
    proj_data_clipped = proj_data.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)
    
    # Rename dimensions to match the forcing data
    proj_data_interp = proj_data_clipped.rename({'x': 'lon', 'y': 'lat'})
    
    # Interpolate onto the grid of forcing data
    proj_data_interp = proj_data_interp.interp(
        lon=forcing_data.lon,
        lat=forcing_data.lat,
        method='linear'
    )
    
    return proj_data_clipped, proj_data_interp



# function to test wether interpolation worked
def plot_proj_data(proj_data_interp, proj_data_clipped, world):

    extent_proj_interp = [
        proj_data_interp.lon.min(),
        proj_data_interp.lon.max(),
        proj_data_interp.lat.min(),
        proj_data_interp.lat.max()
    ]
    
    extent_proj_clipped = [
        proj_data_clipped.x.min(),
        proj_data_clipped.x.max(),
        proj_data_clipped.y.min(),
        proj_data_clipped.y.max()
    ]

    # Create a new figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the raster data and country boundaries for proj_data_interp
    proj_data_interp.max(dim='time').plot(ax=ax1, cmap='coolwarm')
    world.boundary.plot(ax=ax1, linewidth=1.5, color='black')
    ax1.set_extent(extent_proj_interp)
    ax1.set_title('Max Temperature Proj Data Interpolated')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.coastlines()
    ax1.gridlines()

    # Plot the raster data and country boundaries for proj_data_clipped
    proj_data_clipped.max(dim='time').plot(ax=ax2, cmap='coolwarm')
    world.boundary.plot(ax=ax2, linewidth=1.5, color='black')
    ax2.set_extent(extent_proj_clipped)
    ax2.set_title('Max Temperature Proj Data Clipped')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.coastlines()
    ax2.gridlines()

    # Show the plot
    plt.show()


# temporal offest function
def correct_temporal_bias(proj_data, ref_data):
    
    proj_data_current = proj_data.sel(time=slice('2000', '2023'))

    # Calculate the mean for the selected time period
    mean_proj_data_current = proj_data_current.mean(dim='time')
    mean_ref_data = ref_data.mean(dim='time')

    # Calculate the mean difference
    mean_difference = mean_proj_data_current - mean_ref_data

    # Calculate the mean difference over all pixels
    mean_difference_overall = mean_difference.mean().values

    print("Mean Difference Over All Pixels:", mean_difference_overall)

    # Subtract the mean difference from the entire time series
    proj_data_corrected = proj_data - mean_difference_overall

    return proj_data_corrected
