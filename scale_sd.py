import pandas as pd

from prepro import *
from matplotlib import pylab as plt
import seaborn as sns

P_data1 = pd.read_csv('data/total_precipitation/precip_2000_2011.csv')
P_data2 = pd.read_csv('data/total_precipitation/precip_2012_2023.csv')
P_data = pd.concat([P_data1, P_data2])

cutoff = 1e-05

# Scale precipitation standard deviation
P_data = P_data.set_index(['lat', 'lon'])
P_data_double_sd = P_data.copy()
P_data_half_sd = P_data.copy()

# A lot faster with group by
for scale_factor, new_data in zip([2, 0.5], [P_data_double_sd, P_data_half_sd]):
    print('Scale factor', scale_factor)
    for location in tqdm(np.unique(P_data.index)):
        sample = P_data[P_data.index == location]
        scaled_precip = adjust_precipitation(sample['tp'],
                                             scale_factor=scale_factor,
                                             cutoff=cutoff)

        new_data.loc[P_data.index == location, 'tp'] = scaled_precip

# Compare stats
precip_stats = P_data.groupby(['lat', 'lon']).describe()
precip_dbl_stats = P_data_double_sd.groupby(['lat', 'lon']).describe()
precip_half_stats = P_data_half_sd.groupby(['lat', 'lon']).describe()

# Calculate the ratio of standard deviations
ratio_std_dbl = precip_dbl_stats[('tp', 'std')] / precip_stats[('tp', 'std')]
ratio_std_half = precip_half_stats[('tp', 'std')] / precip_stats[('tp', 'std')]
ratio_mean_dbl = precip_dbl_stats[('tp', 'mean')] / precip_stats[('tp', 'mean')]
ratio_mean_half = precip_half_stats[('tp', 'mean')] / precip_stats[
    ('tp', 'mean')]

# Plot resulting mean and std and compare
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].bar(range(len(ratio_std_dbl)), ratio_std_dbl)
ax[0, 1].bar(range(len(ratio_std_half)), ratio_std_half)
ax[1, 0].bar(range(len(ratio_mean_dbl)), ratio_mean_dbl)
ax[1, 1].bar(range(len(ratio_mean_half)), ratio_mean_half)

# Plot a horizontal line
ax[0, 0].axhline(y=2, color='r', linestyle='--', label='target factor')
ax[0, 1].axhline(y=0.5, color='r', linestyle='--')
ax[1, 0].axhline(y=0, color='r', linestyle='--')
ax[1, 1].axhline(y=0, color='r', linestyle='--')

ax[0, 0].set_title('Standard deviation factor (doubled)')
ax[0, 1].set_title('Standard deviation factor (half)')
ax[1, 0].set_title('Mean ratio (doubled)')
ax[1, 1].set_title('Mean ratio (half)')

# Show plot
ax[0, 0].legend()
plt.show()

plt.savefig(f'figs/scaling_{cutoff}_cutoff.pdf')

# Save preprocessed data
path = 'data/total_precipitation/preprocessed'
os.makedirs(path, exist_ok=True)

df_2000_to_2011_doubled, df_2012_to_2023_doubled = split_dataset(
    P_data_double_sd.reset_index())
df_2000_to_2011_half, df_2012_to_2023_half = split_dataset(
    P_data_half_sd.reset_index())

df_2000_to_2011_doubled.to_csv(f'{path}/precip_dbl_2000_2011.csv', index=False)
df_2012_to_2023_doubled.to_csv(f'{path}/precip_dbl_2012_2023.csv', index=False)
df_2000_to_2011_half.to_csv(f'{path}/precip_half_2000_2011.csv', index=False)
df_2012_to_2023_half.to_csv(f'{path}/precip_half_2012_2023.csv', index=False)

# -- analyse scaled precipitation

# Load precipitation data
precip_2000_2011 = pd.read_csv(f'data/total_precipitation/precip_2000_2011.csv')
precip_2012_2023 = pd.read_csv(f'data/total_precipitation/precip_2012_2023.csv')
precip = pd.concat((precip_2000_2011, precip_2012_2023)).reset_index()

scaled_path = 'data/total_precipitation/preprocessed'
precip_scaled = {}
for scale in ['dbl', 'half']:
    # Load scaled data
    precip_scaled_2000_2011 = pd.read_csv(
        f'{scaled_path}/precip_{scale}_2000_2011.csv')
    precip_scaled_2012_2023 = pd.read_csv(
        f'{scaled_path}/precip_{scale}_2012_2023.csv')
    precip_scaled[scale] = pd.concat(
        (precip_scaled_2000_2011, precip_scaled_2012_2023)).reset_index()

# Define the meteorological seasons
seasons = {
    'DJF': [12, 1, 2],  # December, January, February
    'MAM': [3, 4, 5],  # March, April, May
    'JJA': [6, 7, 8],  # June, July, August
    'SON': [9, 10, 11]  # September, October, November
}


# Function to categorize dates into seasons
def get_season(month):
    for season, months in seasons.items():
        if month in months:
            return season


# Add a new column for seasons
precip['season'] = pd.to_datetime(precip['date']).dt.month.map(get_season)
precip_scaled['half']['season'] = pd.to_datetime(
    precip_scaled['half']['date']).dt.month.map(get_season)
precip_scaled['dbl']['season'] = pd.to_datetime(
    precip_scaled['dbl']['date']).dt.month.map(get_season)
precip_scaled['normal'] = precip.copy()

# Plot precipitation distribution per season
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

for season, ax in zip(seasons.keys(), axes.flatten()):
    for k, v in precip_scaled.items():
        data_season = v[v['season'] == season]
        max_value = data_season['tp'].quantile(0.95)
        sns.kdeplot(data=data_season[data_season['tp'] <= max_value], x='tp',
                    ax=ax, fill=False, alpha=0.5, label=k)
        # ax.set_ylim((0, 100))
    ax.set_title(f'Precipitation Distribution - {season}')
    ax.set_xlabel('Precipitation (tp)')
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
plt.show()

plt.savefig('figs/precip_scaled_distribution.pdf')

precip_df = pd.DataFrame({'half': precip_scaled['half']['tp'],
                          'dbl': precip_scaled['dbl']['tp'],
                          'normal': precip_scaled['normal']['tp'],
                          'season': precip_scaled['normal']['season']})

seasonal_stats = np.round(precip_df.groupby('season').describe(), 4)

# Swap level of columns to place 'half', 'double', and 'normal' below each other
seasonal_stats.columns = seasonal_stats.columns.swaplevel(0, 1)

# Sort index to have 'half', 'double', and 'normal' together
seasonal_stats = seasonal_stats.sort_index(axis=1)

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print the seasonal statistics
print(seasonal_stats)

# up to 12% difference in seasonal means
