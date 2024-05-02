import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# File paths
precipitation_file = "/Users/justusnogel/Documents/applied_landsurface_modelling/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.2000.nc"
coordinates_file = "/Users/justusnogel/Documents/applied_landsurface_modelling/catchment_coordinates_donau.txt"

# Open the NetCDF file
nc_file = nc.Dataset(precipitation_file)
lon = nc_file.variables['lon'][:]
lat = nc_file.variables['lat'][:]
data = nc_file.variables['tp'][50, :, :]  # Using arbitrary time index 100

# Create Basemap instance
m = Basemap(llcrnrlon=lon.min(), llcrnrlat=lat.min(),
            urcrnrlon=lon.max(), urcrnrlat=lat.max(),
            projection='cyl', resolution='l')

# Draw coastlines, countries, and states
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# Convert lat/lon values to x/y coordinates for catchment points
catchment_data = np.genfromtxt(coordinates_file, delimiter=',', skip_header=1, dtype=str)
catchment_lat = catchment_data[:, 1].astype(float)
catchment_lon = catchment_data[:, 2].astype(float)
x_catchment, y_catchment = m(catchment_lon, catchment_lat)

# Overlay catchment coordinates in black
plt.scatter(x_catchment, y_catchment, c='black', marker='o', label='Catchment Coordinates')

# Plot the precipitation data as grid cells
x_mesh, y_mesh = np.meshgrid(lon, lat)
x_mesh, y_mesh = m(x_mesh, y_mesh)
plt.pcolormesh(x_mesh, y_mesh, data, cmap='rainbow', alpha=0.5, label='Precipitation Data')

# Add latitude and longitude lines with labels
m.drawparallels(np.arange(-90., 91., 5.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 5.), labels=[0,0,0,1], fontsize=10)

plt.title('Precipitation Data with Catchment Coordinates from Weser (<300 km2)')
plt.legend()

# Show the plot
plt.show()

# Close the NetCDF file
nc_file.close()
