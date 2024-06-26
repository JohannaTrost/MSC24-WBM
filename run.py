import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import itertools
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calc_et_weight(temp, lai, w):
    """Calculate influence of LAI and temperature on ET."""
    # Get coefficients for temperature and lai
    temp_w, lai_w = w

    # Scale data
    data = pd.DataFrame({'temp': temp, 'lai': lai})
    scale = MinMaxScaler()
    scaled_data = pd.DataFrame(scale.fit_transform(data), columns=data.columns)

    # Weight Temperature and LAI
    et_coef = temp_w * scaled_data['temp'] + lai_w * scaled_data['lai']

    return np.asarray(et_coef)


def runoff(wn, Pn, cs, alpha):
    return Pn * (wn / cs) ** alpha


def evapotranspiration(wn, Rn, cs, beta, gamma):
    return beta * (wn / cs) ** gamma * Rn


def snow_function(Snow_n, P_n, T_n,
        c_m):  # Returns 1. Amount of snow after a time step and 2. Amount of water coming into the soil (liquid rain and/or melting snow)
    if T_n <= 273.15:  # Snow stays on the ground
        return Snow_n + P_n, 0
    elif Snow_n < 0.001:  # no accumulated snow and temperature above 0 degrees -> return "0 accumulated snow" and treat precipitation as rain
        return 0, P_n
    else:  # Snow is melting (if there was snow)
        SnowMelt = c_m * (T_n - 273.15)  # Amount of snow melting (if there was)
        if SnowMelt > Snow_n:  # Is the amount of snow that would melt larger than the existing amount of snow?
            return 0, Snow_n + P_n  # no snow remains, all existing snow melts
        else:
            return Snow_n - SnowMelt, SnowMelt + P_n  # Some snow remains, some snow melts


def water_balance(wn, Pn, Rn, Snown, Tn, cs, alpha, beta, gamma, c_m):
    snow, Pn = snow_function(Snown, Pn, Tn,
                             c_m)  # overwrites the precipitation (if snow melts or precipitation is accumulated as snow)
    Qn = runoff(wn, Pn, cs, alpha)
    En = evapotranspiration(wn, Rn, cs, beta, gamma)
    w_next = wn + (Pn - En - Qn)
    w_next = max(0, w_next)
    return Qn, En, w_next, snow


def time_evolution(P_data, R_data, T_data, lai_data, cs, alpha,
        gamma, beta, c_m, et_weight):
    """Calculates the time evolution of the soil moisture, runoff and evapotranspiration.
    Input:  w_0: initial soil moisture [mm]
            P_data: precipitation data [m/day]
            R_data: net radiation data [J/day/m**2]
            Snow_0: initial Snow amount [mm] (equivalent amount of water)
            T_data: temperature []
            cs: Soil water holding capacity [mm]
            alpha: runoff parameter
            beta: evapotranspiration parameter
            gamma: evapotranspiration parameter
            c_m: snow melt parameter [mm/K/day]
            Output: DataFrame with columns: time, Rn, Pr, calculated_soil_moisture, runoff, evapotranspiration"""
    w_0 = 0.9 * cs
    Snow_0 = 0
    conv = 1 / 2260000  # from J/day/m**2 to mm/day
    R_data = R_data * conv
    P_data = P_data * 10 ** 3  # from m/day to mm/day
    output_df = pd.DataFrame(
        columns=['time', 'R', 'P', 'calculated_soil_moisture', 'runoff',
                 'evapotranspiration', 'snow', 'Temperature', 'LAI'])

    # Precompute ET parameter
    et_coefs = beta * calc_et_weight(T_data, lai_data, et_weight)

    for t in range(1, len(P_data) + 1):
        P = P_data[t - 1]
        R = R_data[t - 1]
        T = T_data[t - 1]
        et_coef = et_coefs[t - 1]
        lai = lai_data[t - 1]
        q, e, w, snow = water_balance(w_0, P, R, Snow_0, T, cs, alpha,
                                      et_coef, gamma, c_m)
        output_df.loc[t - 1] = t, R, P, w_0, q, e, snow, T, lai
        w_0 = w
        Snow_0 = snow

    return output_df


def calibration(P_data, R_data, T_data, lai_data, meas_run, calibration_time,
        cs_values, alpha_values, gamma_values, beta_values, cm_values, et_weights):
    """Calibrates the model to the runoff data.
    P_data: list of precipitation data [m/day]
    R_data: list of net radiation data [J/day/m**2]
    meas_run: measured runoff data [mm/day]
    calibration_time: number of years for calibration [years]
    cs_values: list of soil water holding capacity values [mm]
    alpha_values: list of alpha values
    gamma_values: list of gamma values
    beta_values: list of beta values
    cm_values: list of cm values
    Output: best_params: best parameter combination
            best_output_df: DataFrame with columns: time, Rn, Pr, calculated_soil_moisture, runoff, evapotranspiration"""
    P_calibration = np.concatenate(P_data[0:calibration_time])
    R_calibration = np.concatenate(R_data[0:calibration_time])

    parameter_combinations = list(
        itertools.product(cs_values, alpha_values, gamma_values, beta_values,
                          cm_values, et_weights))

    total_runs = len(parameter_combinations)

    correlation_max = 0

    for run_number, params in enumerate(parameter_combinations, start=1):
        output_df = time_evolution(P_calibration, R_calibration,
                                   T_data, lai_data,
                                   *params)

        print(params)
        corr_P, _ = pearsonr(output_df['runoff'] * 100, meas_run['Value'])  ### hier wird noch die falsche Variable genommen
        # print(corr_P)
        if corr_P > correlation_max:
            print(corr_P)
            correlation_max = corr_P
            best_params = params
            #best_output_df = output_df

    return best_params #, best_output_df

def calibration_allcatchments(P_data, R_data, T_data, lai_data, meas_run, calibration_time, parameter_combinations): # add temperature data
    print('start calibration :)')
    '''Calibrates the model to the runoff data.
    P_data: list of precipitation data [m/day]
    R_data: list of net radiation data [J/day/m**2]
    meas_run: measured runoff data [mm/day]
    calibration_time: number of years for calibration [years]
    parameter_combinations: list of parameter combinations
    Output: best_params: best parameter combination
            best_output_df: DataFrame with columns: time, Rn, Pr, calculated_soil_moisture, runoff, evapotranspiration'''

    P_calibration = np.concatenate(P_data[0:calibration_time])
    R_calibration = np.concatenate(R_data[0:calibration_time])
    T_calibration = np.concatenate(T_data[0:calibration_time])
    lai_calibration = np.concatenate(lai_data[0:calibration_time])

    corr = []
    #corr_df['parameters'] = parameter_combinations

    total_runs = len(parameter_combinations)

    for run_number, params in enumerate(parameter_combinations, start=1):
        output_df = time_evolution(P_calibration, R_calibration, T_calibration, lai_calibration, *params)
        corr_P, _ = pearsonr(output_df['runoff'],meas_run['Value'])
        corr.append(corr_P)
        if run_number % 10 == 0:
            print(f'Run {run_number}/{total_runs} done')
            print(corr_P)
    print('calibration done, this was awesome!!')
    #print(corr)
    return corr