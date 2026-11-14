# What it does
# The script adds citywide weather for each 08:00–08:00 shift to data/modeling_dataset.csv
# (from scripts/step4_features.py) and saves data/modeling_dataset_weather.csv.
# Two versions of each weather feature:
#   era5_*  weather that actually happened (ERA5 reanalysis); the "perfect forecast" upper bound
#   fc_*    the forecast that was available when the prediction is made, MIN_LAG shifts ahead
#
# Data: Open-Meteo (https://open-meteo.com), licensed CC BY 4.0.
#   - Historical Weather API, models=era5. ERA5 is a reanalysis on a 0.25° grid (~25 km),
#     not weather-station observations. The nearest grid point is 47.50, -122.25, southeast of
#     downtown. Averaging the 4 points around the city did not bring ERA5 closer to GFS
#     (rain gap 0.90 -> 0.82 mm/shift, temperature MAE worse), so one point is kept.
#   - Previous Runs API, models=gfs_global, variable_previous_dayN = forecast made N×24 h
#     before the valid hour. Seattle has data from 2024-01-21; snowfall data starts
#     2024-12-08 18:00 UTC, so the first complete shift is 2024-12-09.
# Hourly data is cached in data/weather_hourly_*.csv (in git); delete a file to download it again.
#
# Run it from the repository root: uv run python scripts/step8_weather.py

import os

import pandas as pd
import requests


# constant
MIN_LAG = 3                    # keep the same as scripts/step4_features.py
FORECAST_DAY = MIN_LAG         # see "lead time" below
FORECAST_MODEL = 'gfs_global'  # pure GFS; gfs_seamless gives identical day-3 values (HRRR only reaches 48 h)
REANALYSIS_MODEL = 'era5'
FORECAST_START = '2024-01-01'  # Previous Runs archive begins in Jan 2024; earlier hours come back empty
VARIABLES = ['precipitation', 'snowfall', 'temperature_2m', 'wind_gusts_10m']

# lead time: forecast is made when shift t - MIN_LAG ends, i.e. at the start of shift t - (MIN_LAG - 1).
# The target shift's hours are then 24*(MIN_LAG-1) to 24*MIN_LAG - 1 hours ahead, so for every hour
# only forecasts made at least 24*MIN_LAG hours before it were available: previous_day{MIN_LAG}.
assert 1 <= FORECAST_DAY <= 7, 'Previous Runs API only goes back 7 days'

ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
PREVIOUS_RUNS_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
ERA5_CACHE = 'data/weather_hourly_era5.csv'
FORECAST_CACHE = f'data/weather_hourly_{FORECAST_MODEL}_day{FORECAST_DAY}.csv'

# 1. read modeling_dataset.csv
df = pd.read_csv('data/modeling_dataset.csv', parse_dates=['ShiftDate'])

# one point for the whole city: the mean of the 34 station locations
stations = pd.read_csv('data/sfd_stations.csv')
lat, lon = stations['lat'].mean(), stations['lon'].mean()

# hours needed: first shift start to last shift end, plus a day on each side for the UTC offset
first_day = (df['ShiftDate'].min() - pd.Timedelta(days=1)).date()
last_day = (df['ShiftDate'].max() + pd.Timedelta(days=2)).date()


def download(url, cache, model, variables, start):
    """Hourly weather in UTC from Open-Meteo, read from cache if downloaded before."""
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=['time'])
    params = {'latitude': lat, 'longitude': lon, 'hourly': ','.join(variables), 'models': model,
              'start_date': start, 'end_date': str(last_day), 'timezone': 'GMT'}
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()
    body = r.json()
    hourly = pd.DataFrame(body['hourly'])
    hourly['time'] = pd.to_datetime(hourly['time'])
    print(f"downloaded {model}: grid point {body['latitude']:.3f}, {body['longitude']:.3f}, {len(hourly)} hours")
    hourly.to_csv(cache, index=False)
    return hourly


# 2. download or read the hourly weather (UTC)
era5 = download(ARCHIVE_URL, ERA5_CACHE, REANALYSIS_MODEL, VARIABLES, str(first_day))
fc = download(PREVIOUS_RUNS_URL, FORECAST_CACHE, FORECAST_MODEL,
              [f'{v}_previous_day{FORECAST_DAY}' for v in VARIABLES], FORECAST_START)
fc = fc.rename(columns={f'{v}_previous_day{FORECAST_DAY}': v for v in VARIABLES})


def to_shifts(hourly, prefix):
    """Aggregate hourly UTC weather to 08:00–08:00 Seattle shifts."""
    h = hourly.set_index('time')
    # precipitation, snowfall and gusts are totals/maxima over the hour BEFORE the timestamp,
    # so label them by the hour they start; temperature is the value at the timestamp
    period = h[['precipitation', 'snowfall', 'wind_gusts_10m']].shift(-1, freq='h')
    weather = period.join(h['temperature_2m'], how='outer')
    # same rule as cleaning: local wall-clock time minus 8 h (DST days have 23 or 25 hours)
    local = weather.index.tz_localize('UTC').tz_convert('America/Los_Angeles').tz_localize(None)
    shift = (local - pd.Timedelta(hours=8)).normalize()
    g = weather.groupby(shift)
    out = pd.DataFrame({
        'precip_sum': g['precipitation'].sum(min_count=1),     # mm
        'precip_max_hr': g['precipitation'].max(),            # mm in one hour
        'snow_sum': g['snowfall'].sum(min_count=1),           # cm
        'temp_max': g['temperature_2m'].max(),                # °C
        'temp_min': g['temperature_2m'].min(),                # °C
        'gust_max': g['wind_gusts_10m'].max(),                # km/h
    })
    # a shift counts only if every hour is there (e.g. forecast archive starts mid-shift)
    complete = g.count().eq(g.size(), axis=0)
    for col, var in [('precip_sum', 'precipitation'), ('precip_max_hr', 'precipitation'),
                     ('snow_sum', 'snowfall'), ('temp_max', 'temperature_2m'),
                     ('temp_min', 'temperature_2m'), ('gust_max', 'wind_gusts_10m')]:
        out[col] = out[col].where(complete[var])
    out.index.name = 'ShiftDate'
    return out.add_prefix(prefix)


# 3. aggregate to shifts and join on ShiftDate (weather is the same for every station)
# fc_snow_sum is empty before 2024-12-09: snowfall data starts 2024-12-08 18:00 UTC,
# so the first complete shift is 2024-12-09. Left empty, not filled.
shift_weather = to_shifts(era5, 'era5_').join(to_shifts(fc, 'fc_'), how='outer')
df = df.merge(shift_weather, left_on='ShiftDate', right_index=True, how='left')

# 4. save: weather columns go before the target
weather_cols = list(shift_weather.columns)
columns = [c for c in df.columns if c not in weather_cols and c != 'calls'] + weather_cols + ['calls']
df[columns].to_csv('data/modeling_dataset_weather.csv', index=False)

# 5. checks: coverage, and how close the forecast is to what happened
shifts = df.drop_duplicates('ShiftDate').set_index('ShiftDate')[weather_cols]
coverage = shifts.notna().groupby(shifts.index.year).mean().T.round(3)
print('share of shifts with a value, by year:')
print(coverage)
for c in [c for c in weather_cols if c.startswith('fc_')]:
    print(f"first shift with {c}: {shifts[c].first_valid_index().date()}")

print('\nforecast vs ERA5, on shifts with both:')
for name in [c.removeprefix('era5_') for c in weather_cols if c.startswith('era5_')]:
    both = shifts[[f'era5_{name}', f'fc_{name}']].dropna()
    a, f = both[f'era5_{name}'], both[f'fc_{name}']
    print(f"  {name:14s} n={len(both):4d}  corr={a.corr(f):.2f}  mean ERA5={a.mean():6.2f}  "
          f"mean forecast={f.mean():6.2f}  MAE={(a - f).abs().mean():.2f}")
print(f"\nsaved data/modeling_dataset_weather.csv: {len(df)} rows, {len(columns)} columns")
