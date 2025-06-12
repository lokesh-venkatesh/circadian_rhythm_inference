import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from utils import set_seed
from config import INPUT_SIZE, activate_phase_shift, prior_dist_type

os.makedirs('data/processed', exist_ok=True)
os.makedirs('images', exist_ok=True)

set_seed()
df_list = []
for year in range(1970, 2021, 3):
    ds = xr.open_dataset(f'data/raw/phoenix/t2m_{year}_{year+2}/data_stream-oper_stepType-instant.nc')
    df = pd.DataFrame(index=ds['valid_time'], data=ds['t2m'][:, 0, 0])
    df.index = pd.to_datetime(df.index)
    df_list.append(df)

df = pd.concat(df_list)
df.columns = ['Observed']

# convert Kelvin to Celsius
df.Observed = df.Observed - 273.15

df_mst = df.copy()
df_mst.index = df_mst.index.shift(-7, freq='h')
df_mst = df_mst.iloc[7:]

# Calculate the slope of the temperature trend across yearly averages
df_yearly = df_mst.resample('YS').mean()
slope = np.polyfit(df_yearly.index.year, df_yearly['Observed'], 1)[0]

# Add climate adjusted data to dataframe
df_mst['Climate Adjusted'] = df_mst['Observed'] - slope*(df_mst.index.year-2020)

df_yearly = df_mst.resample('YS').mean()
df_yearly.plot()
plt.title('Annual Average Temperature in Phoenix')
plt.ylabel('Temperature (°C)')
plt.xlabel('Year')
plt.gca().get_lines()[1].set_linestyle('--')

# set the color of both lines to orange
plt.gca().get_lines()[0].set_color('orange')
plt.gca().get_lines()[1].set_color('red')

plt.legend()
plt.savefig('images/long_term_trend.png', dpi=300)
plt.close()

if not prior_dist_type=='Seasonal':
    # Compute the week number for each timestamp
    df_mst['weekofyear'] = df_mst.index.isocalendar().week

    # Group by week number and compute climatological weekly averages
    weekly_climatology = df_mst.groupby('weekofyear')['Observed'].mean()

    # Map each timestamp to the corresponding weekly climatological mean
    df_mst['Weekly Mean'] = df_mst['weekofyear'].map(weekly_climatology)

    # Subtract the seasonal cycle
    df_mst['Deseasonalized'] = df_mst['Observed'] - df_mst['Weekly Mean']

    # Plot average of deseasonalized data by week
    df_mst['Deseasonalized'].groupby(df_mst['weekofyear']).mean().plot()

'''
# --- Begin new plotting block for all four stages as subplots (last 10 years only) ---

# Define the time window: last 10 years
last_10_years = df_mst.index >= (df_mst.index.max() - pd.DateOffset(years=10))

fig, axes = plt.subplots(4, 1, figsize=(28, 12), sharex=True)

# 1. Original (Observed)
axes[0].plot(df_mst.index[last_10_years], df_mst['Observed'][last_10_years], color='tab:blue')
axes[0].set_title('Original (Observed) - Last 10 Years')
axes[0].set_ylabel('Temperature (°C)')
axes[0].grid(True)

# 2. After Climate Adjustment
axes[1].plot(df_mst.index[last_10_years], df_mst['Climate Adjusted'][last_10_years], color='tab:orange')
axes[1].set_title('After Climate Adjustment - Last 10 Years')
axes[1].set_ylabel('Temperature (°C)')
axes[1].grid(True)

# 3. After Deseasonalisation
axes[2].plot(df_mst.index[last_10_years], df_mst['Deseasonalized'][last_10_years], color='tab:green')
axes[2].set_title('After Deseasonalisation - Last 10 Years')
axes[2].set_ylabel('Temperature Anomaly (°C)')
axes[2].grid(True)

# 4. After Normalisation
deseasonalized = df_mst['Deseasonalized']
offset = round(deseasonalized.mean(), 8)
scale = round(deseasonalized.std(), 8)
normalised = (deseasonalized - offset) / scale

axes[3].plot(df_mst.index[last_10_years], normalised[last_10_years], color='tab:red')
axes[3].set_title('After Normalisation - Last 10 Years')
axes[3].set_ylabel('Normalised Value')
axes[3].set_xlabel('Date')
axes[3].grid(True)

plt.tight_layout()
plt.savefig('images/all_stages_last10years.png', dpi=300)
plt.close()
# --- End new plotting block ---

df_mst = df_mst.drop(columns=['Climate Adjusted', 'Weekly Mean', 'weekofyear'])
print(df_mst.head())
df_mst = df_mst.rename(columns={'Deseasonalized': 'Climate Adjusted'})
print(df_mst.head())
'''

# Save the processed data
df_mst.to_csv('data/processed/observed_time_series.csv')

offset = round(df_mst['Climate Adjusted'].mean(), 8)
print('offset: ', offset)
scale = round(df_mst['Climate Adjusted'].std(), 8)
print('scale:  ', scale)
dft = (df_mst['Climate Adjusted']-offset)/scale

data_params = pd.Series({'offset': offset, 'scale': scale}, name='values')
data_params.to_csv('data/data_params.csv')

values = dft.values.squeeze()  # (T,) shape if single column
timestamps = dft.index

# Parameters
window_size = INPUT_SIZE  # 64 days
hop_size = int(window_size/4)     # 16 days
total_hours = len(values)

# Generate overlapping windows
X = []
index = []

while True:
    if activate_phase_shift:
        # Random phase shift between -12 and +12 hours
        phase_shift = np.random.randint(-12, 13)
    else:
        phase_shift = 0

    start = len(X) * hop_size
    shifted_start = start + phase_shift
    # Ensure shifted_start is within valid range
    shifted_start = max(0, min(shifted_start, total_hours - window_size))
    end = shifted_start + window_size
    if end > total_hours:
        break
    X.append(values[shifted_start:end])
    index.append(timestamps[shifted_start])
    if not activate_phase_shift and end + hop_size > total_hours:
        break
    if activate_phase_shift and end == total_hours:
        break

X = np.array(X)  # Shape: (num_chunks, window_size)
index = pd.to_datetime(index)

# Create DataFrame with timestamp index
dft_reshaped = pd.DataFrame(data=X, index=index)

# Keep only rows where the index month is January
# dft_reshaped = dft_reshaped[dft_reshaped.index.month.isin([1])]

# Shuffle rows reproducibly
dft_reshaped = dft_reshaped.sample(frac=1)

# Save to CSV
dft_reshaped.to_csv('data/processed/phoenix_64days.csv')