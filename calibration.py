import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import itertools
from scipy.stats import pearsonr
import run

cs_values = [210, 420, 840]
alpha_values = [2, 4, 8]
gamma_values = [0.2, 0.5, 0.8]
beta_values = [0.4, 0.6, 0.8]
cm_values = [1.5, 2, 2.5] 
# define w_0
years = np.arange(2000,2024,1)
P_data = []
R_data = []
T_data = []
calibration_time = [2000,2010]

# define dummy LAI with sinus function
n_time_steps = 
freq = 2 * np.pi / 365  # Frequency of the curve for one year
sinus_curve = .5 * np.sin(freq * np.arange(n_time_steps) + 5)
sinus_curve += .8  # Centered at 0.8
LAI = sinus_curve.copy()

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
    T_data.append(nc_file.variables['air_temperature'][:,0,0])
    nc_file.close()
# get calibration data

# Define the file path
file_path = "C:/Users/User/Documents/AppliedLandsurfaceModeling/Weser/6337521_Q_Day.Cmd.txt"

# Initialize variables to store data
data = []
header_lines_to_skip = 0

# Read the file line by line
with open(file_path, "r") as file:
    for line in file:
        # Skip header lines until "DATA" section
        if line.strip() == "# DATA":
            header_lines_to_skip += 1
            break
        else:
            header_lines_to_skip += 1

# Define column names to read
columns_to_read = ["YYYY-MM-DD", "Value"]

# Read data into DataFrame using pandas, skipping the header lines and selecting specific columns
df = pd.read_csv(file_path, sep=";", skiprows=header_lines_to_skip, skipinitialspace=True, usecols=columns_to_read, encoding='latin1')

# Convert the 'YYYY-MM-DD' column to datetime format
df['YYYY-MM-DD'] = pd.to_datetime(df['YYYY-MM-DD'])

# Filter the DataFrame to include only the data between January 1, 2000, and December 31, 2020
end_date = pd.to_datetime(str(calibration_time[1] - 1) + '-12-31')
start_date = pd.to_datetime(str(calibration_time[0]) + '-01-01')
filtered_df = df[(df['YYYY-MM-DD'] >= start_date) & (df['YYYY-MM-DD'] <= end_date)]
print(filtered_df)

params, best_run = run.calibration(P_data, R_data, filtered_df, calibration_time[1]-calibration_time[0], cs_values, alpha_values, gamma_values, beta_values)

print(params)
plt.plot(best_run['time'][-365:], best_run['runoff'][-365:], label='Runoff')
plt.plot(best_run['time'][-365:], filtered_df['Value'][1:][-365:], label='Measured Runoff')
plt.legend()
plt.ylabel('mm/day')
plt.show()