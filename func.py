import numpy as np
import xarray as xr
import run
from scipy.signal import detrend

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