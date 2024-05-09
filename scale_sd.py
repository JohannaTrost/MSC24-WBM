from prepro import *
from matplotlib import pylab as plt

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
