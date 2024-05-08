import matplotlib.pyplot as plt
import numpy as np


def snow_function(Snow_n, P_n, T_n, c_m):
    snow_stays_mask = T_n > np.ones_like(T_n) * 273.15
    no_snow_mask = Snow_n >= 0.001 * np.ones_like(T_n)

    # -- Calculate snow
    snow_masked = np.ma.array(Snow_n + P_n, mask=snow_stays_mask)
    snow_out_masked = np.ma.array(
        snow_masked.filled(fill_value=np.zeros_like(Snow_n)),
        mask=snow_stays_mask & no_snow_mask,
        fill_value=no_snow_masked.fill_value)

    # melting snow
    SnowMelt = c_m * (T_n - 273.15)

    snow_out = snow_out_masked.filled(fill_value=Snow_n - SnowMelt)

    snow_out[snow_out < 0] = 0

    # -- Calculate water
    water_masked = np.ma.array(np.zeros_like(Snow_n), mask=snow_stays_mask)
    water_out_masked = np.ma.array(
        water_masked.filled(fill_value=P_n),
        mask=snow_stays_mask & no_snow_mask,
        fill_value=no_snow_masked.fill_value)
    water_out = water_out_masked.filled(fill_value=SnowMelt + P_n)

    return snow_out, water_out


def snow_function_old(Snow_n, P_n, T_n,
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


P_n = np.array([[5, 0], [5, 5]])
T_n = np.array([[273 - 2, 273 - 2], [273 + 2, 273 + 5]])
Snow_n = np.array([[0, 5], [0, 5]])
c_m = 0.5

new_snow, new_water = snow_function(Snow_n, P_n, T_n, c_m)

old_water = np.zeros_like(T_n)
old_snow = np.zeros_like(T_n)
for i in range(len(T_n)):
    for j in range(len(T_n[0])):
        old_snow[i, j], old_water[i, j] = snow_function_old(Snow_n[i, j], P_n[i, j], T_n[i, j], c_m)
