import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import itertools
from scipy.stats import pearsonr
import run
import os
#import tqdm
#Only changes needed
folder_number = 1
# define calibration parameter
cs_values = [210, 420, 840]
alpha_values = [2, 4, 8]
gamma_values = [0.2, 0.5, 0.8]
beta_values = [0.4, 0.6, 0.8]
cm_values = [1.5, 2, 2.5]
et_weights = [(.5, .5), (.25, .75), (.75, .25)]
parameter_combinations = list(
    itertools.product(cs_values, alpha_values, gamma_values, beta_values, cm_values, et_weights))
#parameter_combinations = parameter_combinations[0:5]
# define w_0    
years = np.arange(2000,2024,1)

R_data = []
T_data = []
P_data = []
lai_data = []
calibration_time = [2000,2010]
file_path = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.2000.nc' # no need to change
folder_path = 'data/catchment_sample/sample_selection_'+str(folder_number)

nc_file = nc.Dataset(file_path)
lon = nc_file.variables['lon'][:]
lat = nc_file.variables['lat'][:]
nc_file.close()
    # Loop through all files in the folder
counter = 1
for filename in os.listdir(folder_path):
    
    # Init df for current catchment 
    comb_corr_df = pd.DataFrame(columns=['parameters'])
    comb_corr_df['parameters'] = parameter_combinations
    
    data = []
    header_lines_to_skip = 0
    if filename.endswith(".txt"):  # Check if the file is a text file
         file_path = os.path.join(folder_path, filename)
    with open(file_path, "r", encoding='latin1') as file:
        for line in file:
            if line.startswith("# Latitude"):
                latitude = float(line.split(":")[1].strip())
            elif line.startswith("# Longitude"):
                longitude = float(line.split(":")[1].strip())

    a = np.where(abs(lon-longitude) == min(abs(lon-longitude)))[0][0]
    b = np.where(abs(lat-latitude) == min(abs(lat-latitude)))[0][0]
    # now is the time where you could rund a calibration
    with open(file_path, "r", encoding='latin1') as file:
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
    # get radiation, temperature and precipitation data from netCDF files
    for year in years:   
        file_path1 = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.'+str(year)+'.nc'
        nc_file = nc.Dataset(file_path1)
        # 7,8 is the grid cell of interest for the respective catchment area
        dates = nc_file.variables['time'][:]
        P_data.append(nc_file.variables['tp'][:,b,a])
        nc_file.close()
        file_path1 = 'data/net_radiation/nr.daily.calc.era5.0d50_CentralEurope.'+str(year)+'.nc'
        nc_file = nc.Dataset(file_path1)
        #print(nc_file)
        R_data.append(nc_file.variables['nr'][:,b,a])
        nc_file.close()
        file_path1 = 'data/daily_average_temperature/t2m_mean.daily.calc.era5.0d50_CentralEurope.'+str(year)+'.nc'
        nc_file = nc.Dataset(file_path1)
        T_data.append(nc_file.variables['t2m'][:,b,a])
        nc_file.close()  
        file_path1 = 'data/lai/lai.daily.0d50_CentralEurope.'+str(year)+'.nc'
        nc_file = nc.Dataset(file_path1)
        lai_data.append(nc_file.variables['lai'][:,b,a])
        nc_file.close()  

    comb_corr_df[str(latitude), str(longitude)] = run.calibration_allcatchments(P_data, R_data, T_data,lai_data, filtered_df, calibration_time[1]-calibration_time[0], parameter_combinations)
    print(counter, 'catchment(s) done')
    
    file_id = os.path.basename(file_path).split('_')[0]
    
    comb_corr_df.to_csv(
        f'calibration_results/corr_df{folder_number}_{file_id}.csv',
        sep=' ', index=False)

    counter += 1
    
print(comb_corr_df)

print(r"""\

                                   ._ o o
                                   \_`-)|_
                                ,""       \ 
                              ,"  ## |   ಠ ಠ. 
                            ," ##   ,-\__    `.
                          ,"       /     `--._;) -- CONGRATS, YOU DID IT!!!
                        ,"     ## /
                      ,"   ##    /


                """)