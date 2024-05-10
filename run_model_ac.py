from run_ac import *
import os
from tqdm import tqdm
import pandas as pd

years = np.arange(2000, 2024, 1)
params = [420, 8, 0.2, 0.8, 1.5, (0.75, 0.25)]
os.makedirs('calibration_results', exist_ok=True)
longitude_values = np.arange(22)
latitude_values = np.arange(22)
contains_nan = np.zeros((len(latitude_values), len(longitude_values)))
file_path = 'nan_values.txt'
full_data = np.zeros((len(latitude_values), len(longitude_values), 8766, 4))

for i, lat in enumerate(latitude_values):
    for j, lon in tqdm(enumerate(longitude_values)):
        R_data = []
        P_data = []
        T_data = []
        lai_data = []
        # get radiation, temperature and precipitation data from netCDF files
        for year in years:
            file_path1 = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.' + str(
                year) + '.nc'
            nc_file = nc.Dataset(file_path1)
            # 7,8 is the grid cell of interest for the respective catchment area
            dates = nc_file.variables['time'][:]
            P_data.append(nc_file.variables['tp'][:, lat, lon])
            nc_file.close()
            file_path1 = 'data/net_radiation/nr.daily.calc.era5.0d50_CentralEurope.' + str(
                year) + '.nc'
            nc_file = nc.Dataset(file_path1)
            # print(nc_file)
            R_data.append(nc_file.variables['nr'][:, lat, lon])
            nc_file.close()
            file_path1 = 'data/daily_average_temperature/t2m_mean.daily.calc.era5.0d50_CentralEurope.' + str(
                year) + '.nc'
            nc_file = nc.Dataset(file_path1)
            T_data.append(nc_file.variables['t2m'][:, lat, lon])
            nc_file.close()
            file_path1 = 'data/lai/lai.daily.0d50_CentralEurope.' + str(
                year) + '.nc'
            nc_file = nc.Dataset(file_path1)
            lai_data.append(nc_file.variables['lai'][:, lat, lon])
            nc_file.close()
        full_data[i, j, :, 0] = np.concatenate(P_data)
        full_data[i, j, :, 1] = np.concatenate(R_data)
        full_data[i, j, :, 2] = np.concatenate(T_data)
        full_data[i, j, :, 3] = np.concatenate(lai_data)

results = time_evolution(full_data, *params)  # q, e, w, snow
res_xr_ds = out2xarray(results)

# --- Run model with scaled precipiptation variability

# Get timestamps and coordinates from original data
precip_path = 'data/total_precipitation'
nc_file = nc.Dataset(
    f'{precip_path}/tp.daily.calc.era5.0d50_CentralEurope.2005.nc')
lons = nc_file.variables['lon'][:].data
lats = nc_file.variables['lat'][:].data
nc_file.close()

scaled_path = 'data/total_precipitation/preprocessed'
for scale in ['dbl', 'half']:
    # Load scaled data
    precip_2000_2011 = pd.read_csv(f'{scaled_path}/precip_{scale}_2000_2011.csv')
    precip_2012_2023 = pd.read_csv(f'{scaled_path}/precip_{scale}_2012_2023.csv')
    precip = pd.concat((precip_2000_2011, precip_2012_2023)).reset_index()

    # Put loaded scaled data into an arr
    precip_arr = np.ones_like(full_data[:, :, :, 0]) * -1
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            precip_arr[i, j, :] = precip[(precip['lat'] == lat) &
                                         (precip['lon'] == lon)]['tp'].values

    full_data_scaled = full_data.copy()
    full_data_scaled[:, :, :, 0] = precip_arr

    results_scaled = time_evolution(full_data_scaled, *params)
    res_xr_ds = out2xarray(results_scaled)

    # Save results
    out_path = f'results/{scale}_precip_output'
    os.makedirs(out_path, exist_ok=True)
    for out in ['runoff', 'evapotranspiration', 'soil_moisture', 'snow']:
        res_xr_ds[out].to_netcdf(f'{out_path}/{out}.nc')




