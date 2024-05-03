#libs
import pandas as pd
import folium
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

#import data - replce with your filepath my friends
df = pd.read_csv(r"C:\Users\Kilian\Documents\Studium\Freiburg\Applied_Landsurface_Modelling\Output\catchment_coordinates_elbe.txt")

#make string numeric - but now there are nas
df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')
df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')

#drop nas
df = df.dropna()

## interactive map
# Define the coordinates for the bounding box
bbox = [[44.75, 4.75], [55.25, 15.25]]

# Create a map centered at the middle of the bounding box
m = folium.Map(location=[(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2])

# Add a rectangle to the map to represent the bounding box
folium.Rectangle(
    bounds=bbox,
    color='#ff7800',
    fill=True,
    fill_color='#ffff00',
    fill_opacity=0.2
).add_to(m)

# Add points to the map
for lat, lon in zip(df["latitude"], df["longitude"]):
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(m)

# Display the map
m


## static map
# Create a map centered at the middle of the bounding box
bbox = [[44.75, 4.75], [55.25, 15.25]]
m = Basemap(projection='merc', llcrnrlat=bbox[0][0], urcrnrlat=bbox[1][0], llcrnrlon=bbox[0][1], urcrnrlon=bbox[1][1], resolution='i')

m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')

# Add points to the map
x, y = m(df["longitude"].values, df["latitude"].values)
m.plot(x, y, 'bo', markersize=5)

plt.show()