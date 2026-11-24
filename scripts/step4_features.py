# What it does
# The script reads data/station_shift_counts.csv, the cleaned table from 
# notebooks/data_clean.ipynb. That table has one row per station per 08:00–08:00 
# shift, with Triaged Incident and Nurseline/AMR already removed. The script 
# adds the model inputs and saves data/modeling_dataset.csv.
#
# It also saves the inputs for the next FORECAST_DAYS shifts after the last known one
# (calls empty) in data/forecast_features.csv, for the frozen model:
#   - calendar and holiday features come from the date, so they are exact;
#   - the first MIN_LAG shifts keep their real lag features (already known);
#   - later shifts have no recent calls, so each lag feature is filled with its own
#     historical mean for the same station, weekday and month, saved in
#     data/lag_fill_means.csv (built only from shifts known at planning time).

# Run it from the repository root: uv run python scripts/step4_features.py

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday


# constant
MIN_LAG = 3          # forecast 3 shifts ahead; change here if the app needs a different lead time
WEEKS_SAME_DAY = 4   # how many past weeks to average for the same weekday
LONG_WINDOW = 28     # shifts in the long recent average
FORECAST_DAYS = 90   # future shifts written to forecast_features.csv (the horizon tested in the backtest)
MIN_FILL_SHIFTS = 8  # a station x weekday x month mean needs this many shifts, else station x weekday


# 1. read station_shift_counts.csv and sort by station and date
df = pd.read_csv('data/station_shift_counts.csv')
df['ShiftDate'] = pd.to_datetime(df['ShiftDate'])
df = df.sort_values(['Station', 'ShiftDate']).reset_index(drop=True)
 
# 1b. add placeholder rows for the shifts to forecast (calls unknown)
# With lags of at least MIN_LAG, the features are known for the MIN_LAG shifts after
# the last known one; the same code below then fills them in, so training and forecast
# features are built identically. Later shifts, up to FORECAST_DAYS, get their lag
# features from the lookup table in step 5b.
last_known = df['ShiftDate'].max()
future = pd.DataFrame([(station, last_known + pd.Timedelta(days=k))
                       for station in df['Station'].unique()
                       for k in range(1, max(FORECAST_DAYS, MIN_LAG) + 1)],
                      columns=['Station', 'ShiftDate'])
future['calls'] = np.nan
future['Masked'] = False
df = pd.concat([df, future], ignore_index=True).sort_values(['Station', 'ShiftDate']).reset_index(drop=True)
 
# 2. calendar features: weekday, month, year
df['weekday'] = df['ShiftDate'].dt.dayofweek
df['month'] = df['ShiftDate'].dt.month
df['year'] = df['ShiftDate'].dt.year
 
# 3. holiday features: is_holiday, is_holiday_eve
# the federal calendar moves a weekend holiday to the nearest weekday off; calls follow the actual date
# (e.g. July 4 fireworks), so rebuild the same holidays without that move
rules = [Holiday(r.name, month=r.month, day=r.day, offset=r.offset, start_date=r.start_date, end_date=r.end_date)
         for r in USFederalHolidayCalendar().rules]
first, last = df['ShiftDate'].min(), df['ShiftDate'].max() + pd.Timedelta(days=1)
holidays = pd.DatetimeIndex(sorted(set().union(*[r.dates(first, last) for r in rules])))
 
df['is_holiday'] = df['ShiftDate'].isin(holidays).astype(int)
df['is_holiday_eve'] = (df['ShiftDate'] + pd.Timedelta(days=1)).isin(holidays).astype(int)
 
known = df['calls'].where(~df['Masked'])          # outage shifts count as unknown
by_station = known.groupby(df['Station'])
 
def lagged_mean(lags, min_count):
    """Mean of this station's calls over the given lags, if enough of them are known."""
    cols = pd.concat([by_station.shift(k) for k in lags], axis=1)
    return cols.mean(axis=1).where(cols.notna().sum(axis=1) >= min_count)
 
# same-weekday lags that are already known: multiples of 7, at least MIN_LAG
weekly = [7 * w for w in range(1, 10) if 7 * w >= MIN_LAG][:WEEKS_SAME_DAY]
 
# 4. recent call features: lag3, lag7, avg7, avg28, same_weekday_avg4
df[f'lag{MIN_LAG}'] = by_station.shift(MIN_LAG)                                       # latest known shift
df[f'lag{weekly[0]}'] = by_station.shift(weekly[0])                                   # latest known same weekday
df['avg7'] = lagged_mean(range(MIN_LAG, MIN_LAG + 7), min_count=4)                     # 7 shifts from MIN_LAG back
df[f'avg{LONG_WINDOW}'] = lagged_mean(range(MIN_LAG, MIN_LAG + LONG_WINDOW), min_count=LONG_WINDOW // 2)
df[f'same_weekday_avg{WEEKS_SAME_DAY}'] = lagged_mean(weekly, min_count=2)             # same weekday, last few weeks
recent = [f'lag{MIN_LAG}', f'lag{weekly[0]}', 'avg7', f'avg{LONG_WINDOW}', f'same_weekday_avg{WEEKS_SAME_DAY}']
 
# 5. remove masked shifts after calculating lags
df = df[~df['Masked']].copy()
 
columns = ['Station', 'ShiftDate',
           'weekday', 'month', 'year', 'is_holiday', 'is_holiday_eve',
           *recent, 'calls']
is_forecast = df['ShiftDate'] > last_known

# 5b. lag features for shifts more than MIN_LAG after the last known one
# No recent calls exist for them, so each lag feature gets its own mean over the known
# shifts with the same station, weekday and month. Each feature keeps its own pattern:
# lag3 for a Monday is the previous Friday, while avg7 and avg28 hardly depend on the
# weekday. All 5 features are overwritten from shift MIN_LAG + 1 on, because the rolling
# means there would mix a few real shifts with missing ones.
known_rows = df.loc[~is_forecast]
keys = ['Station', 'weekday', 'month']
cell_mean = known_rows.groupby(keys)[recent].mean()
n_shifts = known_rows.groupby(keys).size().rename('n_shifts')
weekday_mean = known_rows.groupby(['Station', 'weekday'])[recent].mean()

far = is_forecast & (df['ShiftDate'] > last_known + pd.Timedelta(days=MIN_LAG))
cell_keys = pd.MultiIndex.from_frame(df.loc[far, keys])
fill = cell_mean.reindex(cell_keys)
# few shifts in a cell (or a station new to that weekday): use the station x weekday mean
thin = n_shifts.reindex(cell_keys).fillna(0).lt(MIN_FILL_SHIFTS).to_numpy() | fill.isna().any(axis=1).to_numpy()
backup = weekday_mean.reindex(pd.MultiIndex.from_frame(df.loc[far, ['Station', 'weekday']])).to_numpy()
df.loc[far, recent] = np.where(thin[:, None], backup, fill.to_numpy())

lookup = cell_mean.join(n_shifts).reset_index()
lookup['history_through'] = last_known.date().isoformat()
lookup.to_csv('data/lag_fill_means.csv', index=False)

# 6. save: known shifts for training, forecast shifts (calls empty) for the app
history = df.loc[~is_forecast, columns].copy()
history['calls'] = history['calls'].astype(int)       # keep calls an integer column
history.to_csv('data/modeling_dataset.csv', index=False)
df.loc[is_forecast, columns].to_csv('data/forecast_features.csv', index=False)
print(history.head())
end = df.loc[is_forecast, 'ShiftDate'].max()
print(f'forecast rows: {is_forecast.sum()} ({(last_known + pd.Timedelta(days=1)).date()} to {end.date()}): '
      f'real lags for the first {MIN_LAG} shifts, lookup means for {far.sum()} rows')
print(f'lag_fill_means.csv: {len(lookup)} cells, {lookup.n_shifts.min()}-{lookup.n_shifts.max()} shifts each, '
      f'{int(thin.sum())} rows used the station x weekday mean, history through {last_known.date()}')
 