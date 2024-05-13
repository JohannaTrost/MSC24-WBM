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
* check precipitation unit (convert kg/km2/s to m/day)
Other things
* test what happens when we take out the daily variation within one month in the forcing data


General advice:
* stay with small pixels until we know what we want to produce
* play around poster is fine, but find a red thread
* find a goal (can be research questions / hypothesis / plot)
