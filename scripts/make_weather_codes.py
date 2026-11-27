# Saves data/weather_codes_gfs_global_day3.csv for the Prediction tab's weather icon: per shift,
# the most notable weather code and mean cloud cover of the same day-3 GFS forecast, hours and
# place as data/weather_hourly_gfs_global_day3.csv, which the model reads and so stays as it is.
# Run from the repository root: uv run python scripts/make_weather_codes.py

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weather_now import CODES_PATH, code_shifts  # noqa: E402
from weather_alerts import PREVIOUS_RUNS_URL, STORED_PATH  # noqa: E402

VARIABLES = ['weather_code_previous_day3', 'cloud_cover_previous_day3']

hours = pd.read_csv(ROOT / STORED_PATH, usecols=['time'], parse_dates=['time'])['time']
stations = pd.read_csv(ROOT / 'data/sfd_stations.csv')
params = {'latitude': stations['lat'].mean(), 'longitude': stations['lon'].mean(),
          'hourly': ','.join(VARIABLES), 'models': 'gfs_global', 'timezone': 'GMT',
          'start_date': str(hours.min().date()), 'end_date': str(hours.max().date())}
r = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=300)
r.raise_for_status()
hourly = pd.DataFrame(r.json()['hourly'])
hourly['time'] = pd.to_datetime(hourly['time'])
hourly = hourly.rename(columns=lambda c: c.removesuffix('_previous_day3'))

shifts = code_shifts(hourly)
shifts['weather_code'] = shifts['weather_code'].astype(int)
shifts.to_csv(ROOT / CODES_PATH, date_format='%Y-%m-%d')
print(f"{len(shifts)} shifts, {shifts.index.min():%Y-%m-%d} to {shifts.index.max():%Y-%m-%d}")
