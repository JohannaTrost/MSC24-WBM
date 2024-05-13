# MSC24-WBM

Final version of the Simple Waterbalance Model (SWBM) including:
* snow implementation
* influence of LAI and Temperature on $\beta_0$

  LAI data:
  * spanning from 2000-2018
  * for the last 5 years we repeat the last 5 existig years
  * missing data for some pixels (water)

To do: <br>
Trends
* normalize variables to show trends in % changes
* test what happens when we only use detrended temperature or radiation --> which one has the higher influence? <br>

CMIP6
* check precipitation unit (convert kg/km2/s to m/day) <br>

Other things
* test what happens when we take out the daily variation within one month in the forcing data


General advice:
* stay with small pixels until we know what we want to produce
* play around poster is fine, but find a red thread
* find a goal (can be research questions / hypothesis / plot)

Text snippets for poster:
* Seasonal changes in Soilmoisture in future climate projection

* Modeling soil moisture is crucial for understanding and predicting ecosystem dynamics, hydrological processes, and climate feedbacks at various temporal and spatial scales.
*  We hypothesize that increasing temperatures in the near future will lead to a decrease in soil moisture levels.

* We modelled future soil moisture using an advanced water balance model incorporating CMIP6 climate projections.
* We ran the model on 484 grid cells with a resolution of 0.5 degrees, covering a geographical area from Longitude 4.75 to 15.74 and Latitude 44.75 to 55.25.
* Compared daily soil moisture values from 2000-2024 to monthly values from 2076-2100.
