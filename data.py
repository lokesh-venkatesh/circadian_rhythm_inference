import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from utils import set_seed
from config import INPUT_SIZE

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

for start in range(0, total_hours - window_size + 1, hop_size):
    end = start + window_size
    X.append(values[start:end])
    index.append(timestamps[start])  # Save the timestamp of the first hour of the chunk

X = np.array(X)  # Shape: (num_chunks, 1536)
index = pd.to_datetime(index)

# Create DataFrame with timestamp index
dft_reshaped = pd.DataFrame(data=X, index=index)

# Shuffle rows reproducibly
dft_reshaped = dft_reshaped.sample(frac=1)

# Save to CSV
dft_reshaped.to_csv('data/processed/phoenix_64days.csv')