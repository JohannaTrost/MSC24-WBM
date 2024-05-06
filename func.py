import numpy as np
import xarray as xr
import run

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