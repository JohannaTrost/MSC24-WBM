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
