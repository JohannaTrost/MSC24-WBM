samples_df = coords_df.sample(n=50, replace=False)
samples_df = samples_df.reset_index(drop=True)
samples_df
