# What it does
# The script reads data/station_shift_counts.csv, the cleaned table from 
# notebooks/data_clean.ipynb. That table has one row per station per 08:00–08:00 
# shift, with Triaged Incident and Nurseline/AMR already removed. The script 
# adds the model inputs and saves data/modeling_dataset.csv.

# Run it from the repository root: uv run python scripts/step4_features.py

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday


# constant
MIN_LAG = 3          # forecast 3 shifts ahead; change here if the app needs a different lead time
WEEKS_SAME_DAY = 4   # how many past weeks to average for the same weekday
LONG_WINDOW = 28     # shifts in the long recent average

# 1. read station_shift_counts.csv and sort by station and date
df = pd.read_csv('data/station_shift_counts.csv')
df['ShiftDate'] = pd.to_datetime(df['ShiftDate'])
df = df.sort_values(['Station', 'ShiftDate']).reset_index(drop=True)

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
df[columns].to_csv('data/modeling_dataset.csv', index=False)
print(df[columns].head())

