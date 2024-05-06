import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap

'Reads the results from the calibration and plots the maximum correlation and the optimal calibration results'

file_path = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.2000.nc' # no need to change
nc_file = nc.Dataset(file_path)
lon = nc_file.variables['lon'][:]
lat = nc_file.variables['lat'][:]
nc_file.close()
calibration_res = np.zeros((len(lon), len(lat)))
# Specify the directory where the calibration result files are located
calibration_dir = 'calibration_results'

# Get a list of all files in the directory
file_list = os.listdir(calibration_dir)

# Initialize the maximum value variable
coordinates = []
max_values = []
max_parameters = []
all_values = np.zeros((len(file_list), 729))
all_parameters = []

# Iterate over each file in the directory
for i, file_name in enumerate(file_list):
    max_value = float('-inf')
    # Construct the full file path
    file_path = os.path.join(calibration_dir, file_name)

    with open(file_path, 'r') as file:  # Open the file for reading
        first_line = next(file).strip()  # Read the first line
        first_line = first_line.replace("parameters ", "")
        first_line = first_line.replace("(", "")
        first_line = first_line.replace(")", "")
        first_line = first_line.replace('"', "")
        coordinate = [float(coord.strip("'")) for coord in first_line.strip("parameters ()").split(', ')]
        for j,line in enumerate(file):  # Iterate over each line in the file
            # Split the line into parameter and value
            parameter, value = line.strip().split(')"')
            
            # Extract the value from the parentheses
            value = float(value)
            all_values[i,j] = value
            if i == 0:
                all_parameters.append(parameter)

            
            # Update the maximum value if the current value is greater
            if value > max_value:
                max_parameter = parameter
                max_value = value

        coordinates.append(coordinate)
        max_parameters.append(max_parameter)
        max_values.append(max_value)
coordinates = np.array(coordinates)
for i in range(len(coordinates[:,0])):
    print(coordinates[i,0], coordinates[i,1])
    a = np.where(abs(lon-coordinates[i,1]) == min(abs(lon-coordinates[i,1])))[0][0]
    b = np.where(abs(lat-coordinates[i,0]) == min(abs(lat-coordinates[i,0])))[0][0]
    print(lat[b], lon[a])
    calibration_res[b, a] = max_values[i]

m = Basemap(llcrnrlon=lon.min()-0.25, llcrnrlat=lat.min()-0.25,
            urcrnrlon=lon.max()+0.25, urcrnrlat=lat.max()+0.25,
            projection='cyl', resolution='l')

m.drawcoastlines()
m.drawcountries()
m.drawstates()

min_lon, min_lat = m(lon.min()-0.25, lat.min()-0.25)
max_lon, max_lat = m(lon.max()+0.25, lat.max()+0.25)
mlon, mlat = m(coordinates[:,1], coordinates[:,0])
qu_lon = [6 , 6, 10, 10]
qu_lat = [47, 50, 50, 47]
mqu_lon, mqu_lat = m(qu_lon, qu_lat)
"""plt.scatter(mqu_lon, mqu_lat)
plt.plot(mqu_lon[0:2],mqu_lat[0:2], color='black')
plt.plot(mqu_lon[1:3],mqu_lat[1:3], color='black')
plt.plot(mqu_lon[2:4],mqu_lat[2:4], color='black')
"""
norm = mcolors.TwoSlopeNorm(vmin=calibration_res.min(), vcenter=0, vmax = calibration_res.max())
plt.imshow(calibration_res, cmap=plt.cm.RdBu,  extent=[min_lon, max_lon, min_lat, max_lat], norm=norm)
plt.plot(mlon, mlat, 'x', color='black')
plt.colorbar()
plt.title('Maximum correlation in each catchment')
plt.savefig('maxcalibration_results.png')
plt.show()
max_index = np.where(np.sum(all_values, axis=0) == max(np.sum(all_values, axis=0)))[0][0]
print(max_index)
print(all_parameters[max_index])
print(all_values[:,max_index])
calibration_res = np.zeros((len(lon), len(lat)))
# Plot the calibration results
for i in range(len(coordinates[:,0])):
    #print(coordinates[i,0], coordinates[i,1])
    a = np.where(abs(lon-coordinates[i,1]) == min(abs(lon-coordinates[i,1])))[0][0]
    b = np.where(abs(lat-coordinates[i,0]) == min(abs(lat-coordinates[i,0])))[0][0]
    #print(lat[b], lon[a])
    print(lat[b]-coordinates[i,0], lon[a]-coordinates[i,1])
    calibration_res[b, a] = all_values[:,max_index][i]
plt.title(all_parameters[max_index])
m.drawcoastlines()
m.drawcountries()
m.drawstates()
norm = mcolors.TwoSlopeNorm(vmin=calibration_res.min(), vcenter=0, vmax = calibration_res.max())
plt.imshow(calibration_res, cmap=plt.cm.RdBu,  extent=[min_lon, max_lon, min_lat, max_lat], norm=norm)
plt.plot(mlon, mlat, 'x', color='black')
plt.colorbar()
plt.savefig('optimalcalibration_results.png')
plt.show()
