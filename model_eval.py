from matplotlib import pyplot as plt
import numpy as np
import netCDF4 as nc
from run import time_evolution

cs_values = [210, 420, 840]
alpha_values = [2, 4, 8]
gamma_values = [0.2, 0.5, 0.8]
beta_values = [0.4, 0.6, 0.8]
cm_values = [1.5, 2, 2.5]
et_weights = [(.5, .5), (.25, .75), (.75, .25)]
# define w_0
years = np.arange(2000,2024,1)
P_data = []
R_data = []
T_data = []
calibration_time = [2000,2010]

# get radiation and precipitation data from netCDF files
for year in years:
    file_path = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.'+str(year)+'.nc'
    nc_file = nc.Dataset(file_path)
    P_data.append(nc_file.variables['tp'][:,0,0])
    dates = nc_file.variables['time'][:]
    nc_file.close()

    file_path = 'data/net_radiation/nr.daily.calc.era5.0d50_CentralEurope.'+str(year)+'.nc'
    nc_file = nc.Dataset(file_path)
    #print(nc_file)
    R_data.append(nc_file.variables['nr'][:,0,0])
    nc_file.close()

    file_path = 'data/daily_average_temperature/t2m_mean.daily.calc.era5.0d50_CentralEurope.'+str(year)+'.nc'
    nc_file = nc.Dataset(file_path)
    T_data.append(nc_file.variables['t2m'][:,0,0])
    nc_file.close()


# define dummy LAI with sinus function
n_time_steps = np.concatenate(T_data).shape[0]
freq = 2 * np.pi / 365  # Frequency of the curve for one year
sinus_curve = .5 * np.sin(freq * np.arange(n_time_steps) + 5)
sinus_curve += .8  # Centered at 0.8
LAI = sinus_curve.copy()

# Flatten data
R_data = np.concatenate(R_data)
P_data = np.concatenate(P_data)
T_data = np.concatenate(T_data)

output = time_evolution(0.9*420, P_data, R_data, 0, T_data, LAI, 420, 2,
        0.4, 0.2, 1.5, (.5, .5))