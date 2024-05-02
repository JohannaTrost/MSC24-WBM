import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import itertools
from scipy.stats import pearsonr

def runoff(wn, Pn, cs, alpha):
    return Pn * (wn / cs)**alpha

def evapotranspiration(wn, Rn, cs, beta, gamma):
    return beta * (wn / cs)**gamma * Rn

def water_balance(wn, Pn, Rn, cs, alpha, beta, gamma):
    Qn = runoff(wn, Pn, cs, alpha)
    En = evapotranspiration(wn, Rn, cs, beta, gamma)
    w_next = wn + (Pn - En - Qn)
    w_next = max(0, w_next)
    return Qn, En, w_next

def time_evolution(w_0, P_data, R_data, cs, alpha, beta, gamma):
    '''Calculates the time evolution of the soil moisture, runoff and evapotranspiration.
    Input:  w_0: initial soil moisture [mm]
            P_data: precipitation data [m/day]
            R_data: net radiation data [J/day/m**2]
            cs: Soil water holding capacity [mm]
            alpha: runoff parameter
            beta: evapotranspiration parameter
            gamma: evapotranspiration parameter
            Output: DataFrame with columns: time, Rn, Pr, calculated_soil_moisture, runoff, evapotranspiration'''
    conv = 1/ 2260000 # from J/day/m**2 to mm/day 
    R_data = R_data * conv 
    P_data = P_data * 10**3 # from m/day to mm/day
    output_df = pd.DataFrame(columns=['time', 'R', 'P', 'calculated_soil_moisture', 'runoff', 'evapotranspiration'])
    for t in range(1,len(P_data)):
        P = P_data[t-1]
        R = R_data[t-1]
        q, e, w = water_balance(w_0, P, R, cs, alpha, beta, gamma)
        output_df.loc[t-1] = t, R, P, w_0, q, e
        w_0 = w
    return(output_df)

def calibration(P_data, R_data, meas_run, calibration_time, cs_values, alpha_values, gamma_values, beta_values):
    '''Calibrates the model to the runoff data.
    P_data: list of precipitation data [m/day]
    R_data: list of net radiation data [J/day/m**2]
    meas_run: measured runoff data [mm/day]
    calibration_time: number of years for calibration [years]
    cs_values: list of soil water holding capacity values [mm]
    alpha_values: list of alpha values
    gamma_values: list of gamma values
    beta_values: list of beta values
    Output: best_params: best parameter combination
            best_output_df: DataFrame with columns: time, Rn, Pr, calculated_soil_moisture, runoff, evapotranspiration'''
    P_calibration = np.concatenate(P_data[0:calibration_time])
    R_calibration = np.concatenate(R_data[0:calibration_time])

    parameter_combinations = list(
        itertools.product(cs_values, alpha_values, gamma_values, beta_values))


    total_runs = len(parameter_combinations)

    correlation_max = 0

    for run_number, params in enumerate(parameter_combinations, start=1):
        w_0 = 0.9 * params[0]
        output_df = time_evolution(w_0, P_calibration, R_calibration, *params)
        print(params)
        corr_P, _ = pearsonr(output_df['runoff']*100,meas_run['Value'][1:])  ### hier wird noch die falsche Variable genommen
        #print(corr_P)
        if corr_P  > correlation_max:
            print(corr_P)
            correlation_max = corr_P
            best_params = params
            best_output_df = output_df

    return best_params, best_output_df
