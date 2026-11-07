# 1. 读取 station_shift_counts.csv，按站点和日期排序
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

# constant
MIN_LAG = 3          # forecast 3 shifts ahead; change here if the app needs a different lead time
WEEKS_SAME_DAY = 4   # how many past weeks to average for the same weekday
LONG_WINDOW = 28     # shifts in the long recent average


df = pd.read_csv('data/station_shift_counts.csv')
df['ShiftDate'] = pd.to_datetime(df['ShiftDate'])
df = df.sort_values(['Station', 'ShiftDate']).reset_index(drop=True)
# 2. 日历特征：weekday, month, year
df['weekday'] = df['ShiftDate'].dt.dayofweek
df['month'] = df['ShiftDate'].dt.month
df['year'] = df['ShiftDate'].dt.year

# 3. 节日特征：is_holiday, is_holiday_eve
holidays = USFederalHolidayCalendar().holidays(df['ShiftDate'].min(), df['ShiftDate'].max() + pd.Timedelta(days=1))
df['is_holiday'] = df['ShiftDate'].isin(holidays).astype(int)
df['is_holiday_eve'] = (df['ShiftDate'] + pd.Timedelta(days=1)).isin(holidays).astype(int)
# 4. 近期电话特征：lag3, lag7, avg7, avg28, same_weekday_avg4
#    （断档的班当作空值，不参与计算）
known = df['calls'].where(~df['Masked'])          # outage shifts count as unknown
by_station = known.groupby(df['Station'])

def lagged_mean(lags, min_count):
    """Mean of this station's calls over the given lags, if enough of them are known."""
    cols = pd.concat([by_station.shift(k) for k in lags], axis=1)
    return cols.mean(axis=1).where(cols.notna().sum(axis=1) >= min_count)

# same-weekday lags that are already known: multiples of 7, at least MIN_LAG
weekly = [7 * w for w in range(1, 10) if 7 * w >= MIN_LAG][:WEEKS_SAME_DAY]

df[f'lag{MIN_LAG}'] = by_station.shift(MIN_LAG)                                       # latest known shift
df[f'lag{weekly[0]}'] = by_station.shift(weekly[0])                                   # latest known same weekday
df['avg7'] = lagged_mean(range(MIN_LAG, MIN_LAG + 7), min_count=4)                     # 7 shifts from MIN_LAG back
df[f'avg{LONG_WINDOW}'] = lagged_mean(range(MIN_LAG, MIN_LAG + LONG_WINDOW), min_count=LONG_WINDOW // 2)
df[f'same_weekday_avg{WEEKS_SAME_DAY}'] = lagged_mean(weekly, min_count=2)             # same weekday, last few weeks
recent = [f'lag{MIN_LAG}', f'lag{weekly[0]}', 'avg7', f'avg{LONG_WINDOW}', f'same_weekday_avg{WEEKS_SAME_DAY}']


# 5. 去掉断档的班（在算完 lag 之后）
df = df[~df['Masked']].copy()

columns = ['Station', 'ShiftDate',
           'weekday', 'month', 'year', 'is_holiday', 'is_holiday_eve',
           *recent, 'calls']
df[columns].to_csv('data/modeling_dataset.csv', index=False)
print(df[columns].head())

