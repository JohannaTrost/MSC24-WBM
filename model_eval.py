from matplotlib import pyplot as plt
import numpy as np
import netCDF4 as nc
from run import time_evolution
from run_basic import time_evolution as time_evolution_basic
import pandas as pd
from scipy.stats import pearsonr

def prepro(raw_data):
    """ Preprocess data for SWBM
    Convert runoff, latent heat flux and solar net radiation to mm.
    Convert time to date.

    :param raw_data: raw input data (pandas df):
         -snr: surface net radiation
         -tp: total precipitation
         -ro: runoff
         -sm: soil moisture at the surface
         -le: latent heat flux
    :return: pre-processed data (pandas df)
    """

    data = {'time': pd.to_datetime(raw_data['time']),
            'lat': raw_data['latitude'],
            'long': raw_data['longitude'],
            'tp': raw_data['tp_[mm]'],
            'sm': raw_data['sm_[m3/m3]'] * 1000,
            'ro': raw_data['ro_[m]'] * 24000,
            'le': raw_data['le_[W/m2]'] * (86400 / 2260000),
            # 86400 (seconds) / 2260000 (latent heat of vaporization
            # of water in J/kg)
            'snr': raw_data['snr_[MJ/m2]'] * (1 / 2.26),
            'temp': raw_data['t2m_[K]']
            }

    return pd.DataFrame(data)

# Test with old data

# Load data
input_swbm_ger = pd.read_csv('data/Data_swbm_Germany_new.csv')
input_ger = prepro(input_swbm_ger)

# define dummy LAI with sinus function
n_time_steps = len(input_ger)
freq = 2 * np.pi / 365  # Frequency of the curve for one year
sinus_curve = .5 * np.sin(freq * np.arange(n_time_steps) + 5)
sinus_curve += .8  # Centered at 0.8
LAI = sinus_curve.copy()

output_final = time_evolution(0.9*420, input_ger['tp'], input_ger['snr'], 0, 
                              input_ger['temp'], LAI, 420, 2, 
                              0.4, 0.2, 1.5, (.5, .5))

output_basic = time_evolution_basic(0.9*420, input_ger['tp'], input_ger['snr'], 420, 2, 0.4, 0.2)

corr_et_final = pearsonr(output_final['evapotranspiration'], input_ger['le']).correlation
corr_et_basic = pearsonr(output_basic['evapotranspiration'], input_ger['le']).correlation

corr_sm_final = pearsonr(output_final['calculated_soil_moisture'], input_ger['sm']).correlation
corr_sm_basic = pearsonr(output_basic['calculated_soil_moisture'], input_ger['sm']).correlation

corr_ro_final = pearsonr(output_final['runoff'], input_ger['ro']).correlation
corr_ro_basic = pearsonr(output_basic['runoff'], input_ger['ro']).correlation

print(pd.DataFrame({'Model': ['Basic', 'Final'], 
                    'ET': [corr_et_basic, corr_et_final],
                    'SM': [corr_sm_basic, corr_sm_final], 
                    'RO': [corr_ro_basic, corr_ro_final]}))

# Test with new data
cs_values = [210, 420, 840]
alpha_values = [2, 4, 8]
gamma_values = [0.2, 0.5, 0.8]
beta_values = [0.4, 0.6, 0.8]
cm_values = [1.5, 2, 2.5]
et_weights = [(.5, .5), (.25, .75), (.75, .25)]
# define w_0
years = np.arange(2000, 2018, 1)
P_data = []
R_data = []
T_data = []
lai_data = []
calibration_time = [2000, 2010]

# get radiation and precipitation data from netCDF files
for year in years:
    file_path = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.' + str(
        year) + '.nc'
    nc_file = nc.Dataset(file_path)
    P_data.append(nc_file.variables['tp'][:, 0, 0])
    dates = nc_file.variables['time'][:]
    nc_file.close()

    file_path = 'data/net_radiation/nr.daily.calc.era5.0d50_CentralEurope.' + str(
        year) + '.nc'
    nc_file = nc.Dataset(file_path)
    # print(nc_file)
    R_data.append(nc_file.variables['nr'][:, 0, 0])
    nc_file.close()

    file_path = 'data/daily_average_temperature/t2m_mean.daily.calc.era5.0d50_CentralEurope.' + str(
        year) + '.nc'
    nc_file = nc.Dataset(file_path)
    T_data.append(nc_file.variables['t2m'][:, 0, 0])
    nc_file.close()

    file_path = 'data/lai/lai.daily.0d50_CentralEurope.' + str(year) + '.nc'
    nc_file = nc.Dataset(file_path)
    lai_data.append(nc_file.variables['lai'][:, 0, 0])
    nc_file.close()

# define dummy LAI with sinus function
# n_time_steps = np.concatenate(T_data).shape[0]
# freq = 2 * np.pi / 365  # Frequency of the curve for one year
# sinus_curve = .5 * np.sin(freq * np.arange(n_time_steps) + 5)
# sinus_curve += .8  # Centered at 0.8
# LAI = sinus_curve.copy()

# Flatten data
R_data = np.concatenate(R_data)
P_data = np.concatenate(P_data)
T_data = np.concatenate(T_data)
lai_data = np.concatenate(lai_data)

output = time_evolution(0.9 * 420, P_data, R_data, 0, T_data, lai_data, 420, 2,
                        0.4, 0.2, 1.5, (.5, .5))

output['evapotranspiration'].plot()
