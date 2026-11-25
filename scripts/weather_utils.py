"""Weather helpers shared by step8_weather.py, step15_weather_extremes.py and the app's weather alerts.

to_shifts    hourly UTC weather -> one row per 08:00-08:00 Seattle shift
events       group consecutive shift dates into events
usual        mean calls on the same weekday +-N weeks, leaving out excluded shifts
"""
import numpy as np
import pandas as pd

# shift column: (source variable, aggregation)
SHIFT_COLUMNS = {
    'precip_sum': ('precipitation', 'sum'),       # mm
    'precip_max_hr': ('precipitation', 'max'),    # mm in one hour
    'snow_sum': ('snowfall', 'sum'),              # cm
    'temp_max': ('temperature_2m', 'max'),        # degC
    'temp_min': ('temperature_2m', 'min'),        # degC
    'gust_max': ('wind_gusts_10m', 'max'),        # km/h
}
# totals/maxima over the hour BEFORE the timestamp; temperature is the value at the timestamp
PERIOD_VARIABLES = ['precipitation', 'snowfall', 'wind_gusts_10m']


def to_shifts(hourly, prefix=''):
    """Aggregate hourly UTC weather (a 'time' column plus any of the 4 variables) to 08:00-08:00 shifts."""
    h = hourly.set_index('time')
    present = [v for v in ['precipitation', 'snowfall', 'wind_gusts_10m', 'temperature_2m'] if v in h]
    # label hourly totals/maxima by the hour they start
    period = h[[v for v in PERIOD_VARIABLES if v in present]].shift(-1, freq='h')
    weather = period.join(h[[v for v in present if v not in PERIOD_VARIABLES]], how='outer')
    # same rule as cleaning: local wall-clock time minus 8 h (DST days have 23 or 25 hours)
    local = weather.index.tz_localize('UTC').tz_convert('America/Los_Angeles').tz_localize(None)
    shift = (local - pd.Timedelta(hours=8)).normalize()
    g = weather.groupby(shift)
    # a shift counts only if every hour is there (e.g. forecast archive starts mid-shift)
    complete = g.count().eq(g.size(), axis=0)
    out = pd.DataFrame({col: (g[var].sum(min_count=1) if how == 'sum' else g[var].agg(how)).where(complete[var])
                        for col, (var, how) in SHIFT_COLUMNS.items() if var in present})
    out.index.name = 'ShiftDate'
    return out.add_prefix(prefix)


def events(dates):
    """Group consecutive shift dates into events; returns a Series date -> event id."""
    d = pd.Series(sorted(dates))
    return pd.Series(((d.diff() != pd.Timedelta(days=1)).cumsum()).to_numpy(), index=d.to_numpy())


def usual(calls, excluded, day, allowed=None, weeks=4):
    """Mean of `calls` (indexed by shift date) on the same weekday +-`weeks` weeks, without the
    shifts marked in `excluded` and, if given, only inside `allowed`; returns (mean, dates used)."""
    window = [day + pd.Timedelta(weeks=k) for k in range(-weeks, weeks + 1) if k != 0]
    window = [w for w in window if w in calls.index and not excluded.get(w, False)]
    if allowed is not None:
        window = [w for w in window if w in allowed]
    return (calls.loc[window].mean() if window else np.nan), window
