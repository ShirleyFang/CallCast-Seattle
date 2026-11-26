# What it does
# Checks how well the GFS forecast (day 3, cached by step8_weather.py) catches the weather the app's
# alerts are about, shift by shift, against ERA5 (what actually happened):
#   strong wind  ERA5 gust_max >= 70 km/h           forecast gust_max >= 65 km/h (Step 15)
#   hard freeze  ERA5 temp_min <= -2 degC           forecast temp_min <= -3 degC
#   cold spell   ERA5 temp_min <= 0 degC on 2+ consecutive shifts
#                                                    forecast temp_min <= -1 degC on 2+ consecutive shifts
# Forecast thresholds = ERA5 threshold + the mean GFS bias (temp_min -1.0 degC, gusts about -4 km/h;
# Step 8), fixed before looking at this table. Prints hits, misses and false alarms, then every
# shift where either side is flagged. Weather only: calls are never read.
#
# Run it from the repository root: uv run python scripts/check_alert_thresholds.py

import pandas as pd

from weather_utils import events, to_shifts

ERA5_CACHE = 'data/weather_hourly_era5.csv'
FORECAST_CACHE = 'data/weather_hourly_gfs_global_day3.csv'
START = '2024-01-21'   # first complete shift with GFS forecasts
END = '2026-01-01'     # exclusive; 2026 weather is added only once the team agrees

# (ERA5 threshold, forecast threshold), fixed in advance
WIND = (70.0, 65.0)         # km/h, gust_max >=
HARD_FREEZE = (-2.0, -3.0)  # degC, temp_min <=
SPELL = (0.0, -1.0)         # degC, temp_min <= on at least SPELL_SHIFTS consecutive shifts
SPELL_SHIFTS = 2

# one row per shift from the hourly caches (every shift, also those with gaps in the call data)
era5 = pd.read_csv(ERA5_CACHE, parse_dates=['time'])
fc = pd.read_csv(FORECAST_CACHE, parse_dates=['time']).rename(columns=lambda c: c.removesuffix('_previous_day3'))
weather = to_shifts(era5, 'era5_').join(to_shifts(fc, 'fc_'), how='inner')
weather = weather[(weather.index >= START) & (weather.index < END)]
assert weather.index.max() < pd.Timestamp('2026-01-01'), '2026 is not used'
assert weather[['era5_gust_max', 'era5_temp_min', 'fc_gust_max', 'fc_temp_min']].notna().all().all()
assert weather.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all(), 'shifts must be consecutive'


def in_spell(cold):
    """True for shifts in a run of at least SPELL_SHIFTS consecutive cold shifts."""
    ev = events(cold.index[cold])
    long_runs = ev.map(ev.value_counts()) >= SPELL_SHIFTS
    return pd.Series(cold.index.isin(long_runs.index[long_runs]), index=cold.index)


observed = pd.DataFrame({
    'strong wind': weather['era5_gust_max'] >= WIND[0],
    'hard freeze': weather['era5_temp_min'] <= HARD_FREEZE[0],
    'cold spell': in_spell(weather['era5_temp_min'] <= SPELL[0]),
})
forecast = pd.DataFrame({
    'strong wind': weather['fc_gust_max'] >= WIND[1],
    'hard freeze': weather['fc_temp_min'] <= HARD_FREEZE[1],
    'cold spell': in_spell(weather['fc_temp_min'] <= SPELL[1]),
})
# the app shows one cold alert for either kind
observed['any cold'] = observed['hard freeze'] | observed['cold spell']
forecast['any cold'] = forecast['hard freeze'] | forecast['cold spell']

print(f"{len(weather)} shifts, {weather.index.min().date()} to {weather.index.max().date()}")
rows = []
for alert in observed.columns:
    o, f = observed[alert], forecast[alert]
    rows.append({'alert': alert, 'observed': o.sum(), 'observed events': events(o.index[o]).nunique(),
                 'hits': (o & f).sum(), 'misses': (o & ~f).sum(), 'false alarms': (~o & f).sum(),
                 'forecast flagged': f.sum(), 'share caught': (o & f).sum() / o.sum() if o.any() else float('nan'),
                 'share false': (~o & f).sum() / f.sum() if f.any() else float('nan')})
print('\nShift level (hits = flagged by both):')
print(pd.DataFrame(rows).round(2).to_string(index=False))

for alert, columns in [('strong wind', ['era5_gust_max', 'fc_gust_max']),
                       ('any cold', ['era5_temp_min', 'fc_temp_min'])]:
    o, f = observed[alert], forecast[alert]
    shown = weather.loc[o | f, columns].round(1)
    shown['result'] = pd.Series('hit', index=shown.index).where(o & f, pd.Series('miss', index=shown.index)
                                                                .where(o, 'false alarm'))
    if alert == 'any cold':
        shown['observed'] = observed.loc[o | f, ['hard freeze', 'cold spell']].apply(
            lambda r: ', '.join(k for k, v in r.items() if v) or '-', axis=1)
        shown['forecast'] = forecast.loc[o | f, ['hard freeze', 'cold spell']].apply(
            lambda r: ', '.join(k for k, v in r.items() if v) or '-', axis=1)
    print(f'\nEvery {alert} shift (observed or forecast):')
    print(shown.to_string())
