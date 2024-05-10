import numpy as np
import netCDF4 as nc
import itertools

import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import xarray as xr


def date_range(start, end):
    # Convert start and end dates to datetime64[ns] objects
    start_date = np.datetime64(start)
    end_date = np.datetime64(end)

    # Generate the date range as datetime64[ns] objects
    date_range = np.arange(start_date, end_date + np.timedelta64(1, 'D'),
                           dtype='datetime64[D]')

    return pd.to_datetime(date_range)


def calc_et_weight(temp, lai, w):
    """Calculate influence of LAI and temperature on ET."""
    # Get coefficients for temperature and lai
    temp_w, lai_w = w
    lai = np.nan_to_num(lai, nan=0)
    temp_min = temp.min(axis=2, keepdims=True)
    temp_max = temp.max(axis=2, keepdims=True)
    lai_min = lai.min(axis=2, keepdims=True)
    lai_max = lai.max(axis=2, keepdims=True)

    # Perform normalization
    normalized_temp = (temp - temp_min) / (temp_max - temp_min)
    normalized_lai = (lai - lai_min) / (lai_max - lai_min)

    # Weight Temperature and LAI
    et_coef = temp_w * normalized_temp + lai_w * normalized_lai
    return et_coef


def runoff(wn, Pn, cs, alpha):
    return Pn * (wn / cs) ** alpha


def evapotranspiration(wn, Rn, cs, beta, gamma):
    return beta * (wn / cs) ** gamma * Rn


'''def snow_function(Snow_n, P_n, T_n,
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
            return Snow_n - SnowMelt, SnowMelt + P_n  # Some snow remains, some snow melts'''


def snow_function(Snow_n, P_n, T_n, c_m):
    snow_stays_mask = T_n > np.ones_like(T_n) * 273.15
    no_snow_mask = Snow_n >= 0.001 * np.ones_like(T_n)

    # -- Calculate snow
    snow_masked = np.ma.array(Snow_n + P_n, mask=snow_stays_mask)
    snow_out_masked = np.ma.array(
        snow_masked.filled(fill_value=np.zeros_like(Snow_n)),
        mask=snow_stays_mask & no_snow_mask,
        fill_value=snow_masked.fill_value)

    # melting snow
    SnowMelt = c_m * (T_n - 273.15)

    snow_out = snow_out_masked.filled(fill_value=Snow_n - SnowMelt)

    snow_out[snow_out < 0] = 0

    # -- Calculate water
    water_masked = np.ma.array(np.zeros_like(Snow_n), mask=snow_stays_mask)
    water_out_masked = np.ma.array(
        water_masked.filled(fill_value=P_n),
        mask=snow_stays_mask & no_snow_mask,
        fill_value=snow_masked.fill_value)
    water_out = water_out_masked.filled(fill_value=SnowMelt + P_n)

    return snow_out, water_out


def water_balance(wn, Pn, Rn, Snown, Tn, cs, alpha, beta, gamma, c_m):
    snow, Pn = snow_function(Snown, Pn, Tn,
                             c_m)  # overwrites the precipitation (if snow melts or precipitation is accumulated as snow)
    Qn = runoff(wn, Pn, cs, alpha)
    En = evapotranspiration(wn, Rn, cs, beta, gamma)
    w_next = wn + (Pn - En - Qn)
    w_next = np.maximum(0, w_next)
    """if np.isnan(Qn).any():
        print('Qn contains NaN values')
    if np.isnan(En).any():
        print('En contains NaN values')
    if np.isnan(w_next).any():
        print('w_next contains NaN values')
    if np.isnan(snow).any():
        print('snow contains NaN values')"""

    return Qn, En, w_next, snow


def out2xarray(output, start_year=2000, end_year=2024):
    output = np.moveaxis(output, 2, 0)  # move time axis to be first axis

    # get dates and coordinates
    times = date_range('2000-01-01', '2023-12-31')
    nc_file = nc.Dataset(f'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.2005.nc')
    lons = nc_file.variables['lon'][:].data
    lats = nc_file.variables['lat'][:].data
    nc_file.close()

    out_dict = {}
    for i, out_name in enumerate(['runoff',
                                  'evapotranspiration',
                                  'soil_moisture',
                                  'snow']):
        out_xr = xr.DataArray(output[:, :, :, i], dims=('time', 'lat', 'lon'),
                              coords={'time': times,
                                      'lat': lats,
                                      'lon': lons})
        out_dict[out_name] = out_xr

    return xr.Dataset(out_dict)


def time_evolution(full_data, cs, alpha, gamma, beta, c_m, et_weight):
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
    P_data = full_data[:, :, :, 0]
    R_data = full_data[:, :, :, 1]
    T_data = full_data[:, :, :, 2]
    lai_data = full_data[:, :, :, 3]

    w_0 = 0.9 * cs * np.ones_like((P_data[:, :, 0]))
    Snow_0 = np.zeros_like((P_data[:, :, 0]))
    conv = 1 / 2260000  # from J/day/m**2 to mm/day
    R_data = R_data * conv
    P_data = P_data * 10 ** 3  # from m/day to mm/day
    output = np.zeros(
        (len(P_data[:, 0, 0]), len(P_data[0, :, 0]), len(P_data[0, 0, :]), 4))

    # Precompute ET parameter
    et_coefs = beta * calc_et_weight(T_data, lai_data, et_weight)
    print('start_timeevolution')
    # for t in range(1, len(P_data[0,0,:]) + 1):
    for t in tqdm(range(1, len(P_data[0, 0, :]) + 1)):

        P = P_data[:, :, t - 1]
        R = R_data[:, :, t - 1]
        T = T_data[:, :, t - 1]
        et_coef = et_coefs[:, :, t - 1]

        q, e, w, snow = water_balance(w_0, P, R, Snow_0, T, cs, alpha,
                                      et_coef, gamma, c_m)
        output[:, :, t - 1, 0] = q
        output[:, :, t - 1, 1] = e
        output[:, :, t - 1, 2] = w
        output[:, :, t - 1, 3] = snow

        w_0 = w
        Snow_0 = snow

    return output


def calibration(P_data, R_data, T_data, lai_data, meas_run, calibration_time,
        cs_values, alpha_values, gamma_values, beta_values, cm_values,
        et_weights):
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
        corr_P, _ = pearsonr(output_df['runoff'] * 100, meas_run[
            'Value'])  ### hier wird noch die falsche Variable genommen
        # print(corr_P)
        if corr_P > correlation_max:
            print(corr_P)
            correlation_max = corr_P
            best_params = params
            # best_output_df = output_df

    return best_params  # , best_output_df


def calibration_allcatchments(P_data, R_data, T_data, lai_data, meas_run,
        calibration_time, parameter_combinations):  # add temperature data
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
    # corr_df['parameters'] = parameter_combinations

    total_runs = len(parameter_combinations)

    for run_number, params in enumerate(parameter_combinations, start=1):
        output_df = time_evolution(P_calibration, R_calibration, T_calibration,
                                   lai_calibration, *params)
        corr_P, _ = pearsonr(output_df['runoff'], meas_run['Value'])
        corr.append(corr_P)
        if run_number % 10 == 0:
            print(f'Run {run_number}/{total_runs} done')
            print(corr_P)
    print('calibration done, this was awesome!!')
    # print(corr)
    return corr
