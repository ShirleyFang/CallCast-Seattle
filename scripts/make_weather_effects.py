# What it does
# Saves how citywide calls changed with the weather, for the app's Explore > Weather tab:
#   data/weather_effects.csv        chart 1 (temperature) and chart 2 (day of a cold spell): one row
#                                   per group, with shifts, events, usual calls, calls, extra calls, extra %
#   data/weather_events.csv         chart 4: a few past weather events checked in ERA5, one row each
#   data/weather_event_shifts.csv   chart 4: every shift from a week before to a week after each event
#
# Same method as Step 15 (b): usual = citywide calls on the same weekday +-4 weeks, leaving out
# extreme-weather shifts. Here holidays and special days (is_holiday, is_holiday_eve, Dec 24-26,
# Dec 31, Black Friday) are also left out of charts 1 and 2 and of every usual, so a cold
# Christmas week does not look like a weather effect. Chart 4 keeps every shift and says so.
# Weather is ERA5 at one point for the city (step8_weather.py). Calls 2022-2025; 2026 is not used.
#
# Run it from the repository root: uv run python scripts/make_weather_effects.py

import numpy as np
import pandas as pd

from cv_utils import DATE_COL, TARGET, load_dataset
from feature_sets import SPECIAL_DAYS, add_special_days
from weather_utils import events, to_shifts, usual

DATA_PATH = 'data/modeling_dataset.csv'
ERA5_CACHE = 'data/weather_hourly_era5.csv'
EFFECTS_PATH = 'data/weather_effects.csv'
EVENTS_PATH = 'data/weather_events.csv'
EVENT_SHIFTS_PATH = 'data/weather_event_shifts.csv'
START, END = '2022-01-01', '2026-01-01'   # END exclusive: 2026 is the test year
USUAL_WEEKS = 4
WINDOW_DAYS = 7                           # chart 4: days shown before and after each event

# extreme shifts left out of "usual" (Step 15 thresholds, plus heat), fixed before looking
WIND, FREEZE, SNOW, HEAT = 70.0, 0.0, 1.0, 30.0   # gust >= km/h, low <= degC, snow >= cm, high >= degC
SPELL_SHIFTS = 2                                  # a cold spell: 2+ consecutive freezing shifts

# chart 1: each shift goes to the first group it matches (cold side by the low, hot side by the high)
TEMPERATURE_GROUPS = [
    ('Low -2 °C or colder', lambda w: w['temp_min'] <= -2),
    ('Low -2 to 0 °C', lambda w: w['temp_min'] <= 0),
    ('Low 0 to 2 °C', lambda w: w['temp_min'] <= 2),
    ('High 30 °C or hotter', lambda w: w['temp_max'] >= 30),
    ('High 25 to 30 °C', lambda w: w['temp_max'] >= 25),
    ('Mild', lambda w: pd.Series(True, index=w.index)),
]

# chart 4: dates checked in ERA5 below (assert), described in plain words
WEATHER_EVENTS = [
    {'event': 'Dec 2022 cold, snow and ice', 'kind': 'cold', 'start': '2022-12-20', 'end': '2022-12-23',
     'note': 'Snow and lows of -5 °C, then freezing rain. Christmas week, so part of the change may be the holiday.'},
    {'event': 'Jan 2024 cold snap', 'kind': 'cold', 'start': '2024-01-11', 'end': '2024-01-16',
     'note': 'The coldest spell in the data: six days with lows of about -5 to -10 °C. Includes Martin Luther King Jr. Day.'},
    {'event': 'Jul 2022 heat wave', 'kind': 'heat', 'start': '2022-07-26', 'end': '2022-07-31',
     'note': 'Six days in a row above 30 °C.'},
    {'event': 'Nov 2024 windstorm', 'kind': 'wind', 'start': '2024-11-19', 'end': '2024-11-19',
     'note': 'A strong storm with gusts near 80 km/h.'},
]

# 1. citywide calls per shift (outage shifts are already left out), holidays, ERA5 weather
df = load_dataset(DATA_PATH)
df = df[(df[DATE_COL] >= START) & (df[DATE_COL] < END)]
assert df[DATE_COL].max() < pd.Timestamp('2026-01-01'), '2026 is not used'
df, _ = add_special_days(df)
holiday_cols = ['is_holiday', 'is_holiday_eve', *SPECIAL_DAYS]
city = df.groupby(DATE_COL).agg(calls=(TARGET, 'sum'), stations=(TARGET, 'size'),
                                **{c: (c, 'max') for c in holiday_cols})
assert city['stations'].eq(34).all(), 'every kept shift has all 34 stations'
weather = to_shifts(pd.read_csv(ERA5_CACHE, parse_dates=['time']))
city = city.join(weather[['temp_min', 'temp_max', 'gust_max', 'snow_sum']], how='inner')
assert city[['temp_min', 'temp_max', 'gust_max', 'snow_sum']].notna().all().all()
holiday = city[holiday_cols].eq(1).any(axis=1)
extreme = ((city['gust_max'] >= WIND) | (city['temp_min'] <= FREEZE) | (city['snow_sum'] >= SNOW)
           | (city['temp_max'] >= HEAT))
calls = city['calls'].astype(float)
city['usual'] = [usual(calls, extreme | holiday, d, weeks=USUAL_WEEKS)[0] for d in city.index]
print(f"{len(city)} shifts, {city.index.min().date()} to {city.index.max().date()}; "
      f"{holiday.sum()} holiday shifts and {extreme.sum()} extreme-weather shifts left out of usual")


def summary(chart, group, order, days):
    """One row: how calls on `days` compare with usual (mean of the per-shift differences)."""
    sub = city.loc[days].dropna(subset=['usual'])
    return {'chart': chart, 'group': group, 'order': order, 'shifts': len(sub),
            'events': events(sub.index).nunique() if len(sub) else 0,
            'usual_calls': round(sub['usual'].mean(), 1), 'calls': round(sub['calls'].mean(), 1),
            'extra_calls': round((sub['calls'] - sub['usual']).mean(), 1),
            'extra_pct': round(100 * (sub['calls'] / sub['usual'] - 1).mean(), 1), 'years': '2022-2025'}


# 2. chart 1: calls by temperature (holidays left out)
rows = []
left = city.index[~holiday]
for order, (group, rule) in enumerate(TEMPERATURE_GROUPS):
    match = left[rule(city.loc[left]).to_numpy()]
    rows.append(summary('temperature', group, order, match))
    left = left.difference(match)
assert left.empty

# 3. chart 2: day of a cold spell (runs of 2+ freezing shifts), and single freezing shifts
freezing = city['temp_min'] <= FREEZE
run = events(city.index[freezing])
length = run.map(run.value_counts())
day_in_run = run.groupby(run).cumcount() + 1
spell_days = {'Single cold day': length.index[length < SPELL_SHIFTS],
              'Day 1 of a cold spell': day_in_run.index[(length >= SPELL_SHIFTS) & (day_in_run == 1)],
              'Day 2': day_in_run.index[(length >= SPELL_SHIFTS) & (day_in_run == 2)],
              'Day 3': day_in_run.index[(length >= SPELL_SHIFTS) & (day_in_run == 3)],
              'Day 4 or later': day_in_run.index[(length >= SPELL_SHIFTS) & (day_in_run >= 4)]}
for order, (group, days) in enumerate(spell_days.items()):
    days = days[~holiday.loc[days].to_numpy()]
    row = summary('cold spell', group, order, days)
    row['events'] = run.loc[days].nunique()   # spells, not runs of these days
    rows.append(row)
effects = pd.DataFrame(rows)
effects.to_csv(EFFECTS_PATH, index=False)
print('\n' + effects.to_string(index=False))

# 4. chart 4: past events, checked in ERA5, with a week on each side
CHECKS = {'cold': lambda w: w['temp_min'] <= FREEZE, 'heat': lambda w: w['temp_max'] >= HEAT,
          'wind': lambda w: w['gust_max'] >= WIND}
event_rows, shift_rows = [], []
for e in WEATHER_EVENTS:
    days = pd.date_range(e['start'], e['end'])
    assert CHECKS[e['kind']](city.loc[days]).all(), f"{e['event']}: ERA5 does not show it on every day"
    window = pd.date_range(days[0] - pd.Timedelta(days=WINDOW_DAYS), days[-1] + pd.Timedelta(days=WINDOW_DAYS))
    sub = city.loc[city.index.intersection(window)]
    inside = sub.index.isin(days)
    event_rows.append({**e, 'shifts': len(days), 'lowest_low': city.loc[days, 'temp_min'].min().round(1),
                       'highest_high': city.loc[days, 'temp_max'].max().round(1),
                       'strongest_gust': city.loc[days, 'gust_max'].max().round(1),
                       'snow_cm': city.loc[days, 'snow_sum'].sum().round(1),
                       'usual_calls': round(sub.loc[inside, 'usual'].mean(), 1),
                       'calls': round(sub.loc[inside, 'calls'].mean(), 1),
                       'extra_pct': round(100 * (sub.loc[inside, 'calls'] / sub.loc[inside, 'usual'] - 1).mean(), 1),
                       'holiday_shifts': int(holiday.loc[days].sum())})
    shift_rows.append(pd.DataFrame({'event': e['event'], 'ShiftDate': sub.index.date, 'in_event': inside,
                                    'calls': sub['calls'].to_numpy(), 'usual_calls': sub['usual'].round(1).to_numpy(),
                                    'temp_min': sub['temp_min'].round(1).to_numpy(),
                                    'temp_max': sub['temp_max'].round(1).to_numpy(),
                                    'gust_max': sub['gust_max'].round(1).to_numpy(),
                                    'holiday': holiday.loc[sub.index].to_numpy()}))
pd.DataFrame(event_rows).to_csv(EVENTS_PATH, index=False)
pd.concat(shift_rows).to_csv(EVENT_SHIFTS_PATH, index=False)
print('\n' + pd.DataFrame(event_rows).drop(columns='note').to_string(index=False))
print(f"\nsaved {EFFECTS_PATH}, {EVENTS_PATH}, {EVENT_SHIFTS_PATH}")
