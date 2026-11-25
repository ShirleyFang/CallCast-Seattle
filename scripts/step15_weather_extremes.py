# What it does
# Rare extreme weather (strong wind, freezing, snow): can the forecast see it, how big is the
# effect on calls, and does any way of using it beat the no-weather model?
#   a. forecast skill: GFS forecast vs ERA5 for strong wind, freezing and snow (2024-01-21 to 2025)
#   b. effect size: calls on extreme shifts vs "usual" (same weekday, +-4 weeks), bootstrap by storm event
#   c. 2025 rolling CV (cv_utils folds and metrics, same model as step 9), all compared with A:
#        F      forecast flags (0/1); trained on ERA5 flags, validated on forecast flags
#        F'     forecast flags in training and validation, training from 2024-01-21 (compare with A')
#        W5/W20 as F, extreme training shifts weighted 5x / 20x
#        R      A's predictions x r on shifts with a strong-wind forecast; r from training shifts only
#   d. ERA5 strong-wind shift counts for 2003-2019 (2020-21 left out): would more history help?
# 2026 is the test set and is dropped. Results are printed; only the 2003-2019 counts are saved.
#
# Run it from the repository root: uv run python scripts/step15_weather_extremes.py
# Robustness check with the day-4 forecast (after step8_weather.py 4): uv run python scripts/step15_weather_extremes.py 4

import os
import sys

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

from cv_utils import DATE_COL, STATION_COL, TARGET, build_features, iter_folds, load_dataset, metrics
from feature_sets import BASELINE_NUMERIC


# constant: model and data (same as step 9)
MODEL_PARAMS = {'loss': 'poisson', 'max_iter': 200, 'random_state': 0}
FORECAST_DAY = int(sys.argv[1]) if len(sys.argv) > 1 else 3    # previous_dayN from step 8; 3 is the main run
DATA_PATH = 'data/modeling_dataset_weather.csv' if FORECAST_DAY == 3 else f'data/modeling_dataset_weather_day{FORECAST_DAY}.csv'
TEST_START = '2026-01-01'          # 2026 is never used
FC_START = {3: '2024-01-21', 4: '2024-01-22'}[FORECAST_DAY]   # first complete shift with GFS forecasts
FC_SNOW_START = '2024-12-09'       # first complete shift with GFS snowfall
EXCLUDE = ['year']

# thresholds, fixed before looking at any results
WIND_ERA5 = 70.0     # km/h, ERA5 gust_max: "strong wind"
WIND_FC = 65.0       # km/h, forecast gust_max: GFS is ~4 km/h weaker than ERA5 on average (step 8),
                     # so 70 - 4 rounded down. The hit/miss table in (a) was seen while exploring,
                     # but was not used to pick this value.
FREEZE = 0.0         # degC, temp_min <= FREEZE
# (b) sensitivity checks for freezing, also fixed before looking at their results
FREEZE_HARD = -2.0   # degC, a stricter freezing threshold
SPELL_SHIFTS = 2     # a cold spell = at least this many consecutive freezing shifts
SNOW = 1.0           # cm, snow_sum >= SNOW
USUAL_WEEKS = 4      # "usual" = same weekday, up to 4 weeks before and after
WEIGHTS = [5, 20]    # sample weights for extreme training shifts, fixed in advance
N_BOOT = 2000
SEED = 0

# 2003-2019 history (d): ERA5 only, no calls; only the yearly counts are saved
HISTORY_START, HISTORY_END = '2003-01-01', '2019-12-31'
HISTORY_COUNTS = 'data/era5_strong_wind_counts_2003_2019.csv'
ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'

WEATHER = ['precip_sum', 'precip_max_hr', 'snow_sum', 'temp_max', 'temp_min', 'gust_max']
SOURCE_COLS = [f'{s}_{w}' for s in ('era5', 'fc') for w in WEATHER]
FLAGS = ['flag_wind', 'flag_freeze', 'flag_snow']

pd.set_option('display.width', 200)

# 1. read the dataset and drop 2026
df = load_dataset(DATA_PATH)
df = df[df[DATE_COL] < TEST_START]
assert df[DATE_COL].max() < pd.Timestamp(TEST_START)
print(f"{len(df)} rows, {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
print("Note: A is the baseline model. The final model also has the special days (is_nye, "
      "is_day_after_thanksgiving, is_xmas), built in memory by train_model.py, not stored in this dataset.")

# one row per shift: citywide calls and the (citywide) weather
city = df.groupby(DATE_COL).agg(calls=(TARGET, 'sum'), stations=(TARGET, 'size'),
                                **{c: (c, 'first') for c in SOURCE_COLS})


def flags(source):
    """0/1 extreme-weather flags per shift from 'era5' or 'fc' (empty where the source is empty)."""
    wind = WIND_ERA5 if source == 'era5' else WIND_FC
    g, t, s = city[f'{source}_gust_max'], city[f'{source}_temp_min'], city[f'{source}_snow_sum']
    return pd.DataFrame({'flag_wind': (g >= wind).astype(float).where(g.notna()),
                         'flag_freeze': (t <= FREEZE).astype(float).where(t.notna()),
                         'flag_snow': (s >= SNOW).astype(float).where(s.notna())})


shift_flags = {'era5': flags('era5'), 'fc': flags('fc')}
era5_any = shift_flags['era5'].eq(1).any(axis=1)            # any ERA5 extreme: left out of "usual"


def events(dates):
    """Group consecutive shift dates into events; returns a Series date -> event id."""
    d = pd.Series(sorted(dates))
    return pd.Series(((d.diff() != pd.Timedelta(days=1)).cumsum()).to_numpy(), index=d.to_numpy())


def usual(day, allowed=None):
    """Mean citywide calls on the same weekday +-USUAL_WEEKS weeks, not extreme, inside `allowed`."""
    window = [day + pd.Timedelta(weeks=k) for k in range(-USUAL_WEEKS, USUAL_WEEKS + 1) if k != 0]
    window = [w for w in window if w in city.index and not era5_any.get(w, False)]
    if allowed is not None:
        window = [w for w in window if w in allowed]
    return (city.loc[window, 'calls'].mean() if window else np.nan), window


# 2. (a) forecast skill: shift-level hits / misses / false alarms
print('\n=== (a) Can the forecast see it? (shift level) ===')
fc_period = city[city.index >= FC_START]
rows = []
for label, era5_col, fc_col, op, obs_t, fc_ts, start in [
        ('strong wind', 'era5_gust_max', 'fc_gust_max', 'ge', WIND_ERA5, [55, 60, 65, 70], FC_START),
        ('freezing', 'era5_temp_min', 'fc_temp_min', 'le', FREEZE, [FREEZE], FC_START),
        ('snow', 'era5_snow_sum', 'fc_snow_sum', 'ge', SNOW, [SNOW], FC_SNOW_START)]:
    for period, part in [('2024', fc_period[fc_period.index.year == 2024]),
                         ('2025', fc_period[fc_period.index.year == 2025]), ('both', fc_period)]:
        part = part[part.index >= start].dropna(subset=[era5_col, fc_col])
        obs = getattr(part[era5_col], op)(obs_t)
        for t in fc_ts:
            pred = getattr(part[fc_col], op)(t)
            hits = int((pred & obs).sum())
            rows.append({'event': label, 'period': period, 'forecast threshold': t, 'shifts': len(part),
                         'observed': int(obs.sum()), 'hits': hits, 'misses': int((~pred & obs).sum()),
                         'false alarms': int((pred & ~obs).sum()),
                         'hit rate': hits / obs.sum() if obs.sum() else np.nan,
                         'false alarm ratio': (pred & ~obs).sum() / pred.sum() if pred.sum() else np.nan})
print(pd.DataFrame(rows).round(2).to_string(index=False))
wind_shifts = fc_period[(fc_period['era5_gust_max'] >= WIND_ERA5) | (fc_period['fc_gust_max'] >= WIND_FC)]
print(f'\nShifts with ERA5 gust >= {WIND_ERA5:.0f} or forecast gust >= {WIND_FC:.0f} (km/h):')
print(wind_shifts[['era5_gust_max', 'fc_gust_max', 'calls']].round(1).to_string())

# 3. (b) effect size: ratio of calls to "usual", bootstrap over events
print('\n=== (b) Effect size: calls / usual (same weekday, +-4 weeks, non-extreme shifts) ===')
rng = np.random.default_rng(SEED)


def effect(label, mask, period, end):
    """Mean calls / usual on the masked shifts before `end`, usual shifts also before `end`; event bootstrap CI."""
    allowed = set(city.index[city.index < end])
    days = city.index[mask & (city.index < end)]
    ratio = pd.Series({d: city.loc[d, 'calls'] / usual(d, allowed)[0] for d in days}).dropna()
    ev = events(ratio.index)
    groups = [ratio[ev == e].to_numpy() for e in ev.unique()]
    boot = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean()
            for _ in range(N_BOOT)]
    return {'event': label, 'period': period, 'shifts': len(ratio), 'events': len(groups),
            'mean ratio': ratio.mean(), 'CI low': np.percentile(boot, 2.5), 'CI high': np.percentile(boot, 97.5)}


effect_rows = []
for label, mask in [(f'strong wind (ERA5 gust >= {WIND_ERA5:.0f})', shift_flags['era5']['flag_wind'].eq(1)),
                    (f'freezing (ERA5 temp_min <= {FREEZE:.0f})', shift_flags['era5']['flag_freeze'].eq(1)),
                    (f'snow (ERA5 snow_sum >= {SNOW:.0f} cm)', shift_flags['era5']['flag_snow'].eq(1))]:
    for period, end in [('2022-2025', TEST_START), ('2022-2024', '2025-01-01')]:
        effect_rows.append(effect(label, mask, period, end))
print(pd.DataFrame(effect_rows).round(3).to_string(index=False))
print('Ratio 1.10 = 10% more calls than usual. 95% CI from resampling whole events (consecutive '
      'extreme shifts = one event). Usual excludes any ERA5-extreme shift and stays inside the period.')

# sensitivity for freezing (2022-2025): a stricter threshold, and cold spells vs single freezing shifts.
# "usual" is unchanged (it still leaves out every shift with temp_min <= FREEZE).
freeze = shift_flags['era5']['flag_freeze'].eq(1)
run_length = events(city.index[freeze]).pipe(lambda ev: ev.map(ev.value_counts()))
in_spell = pd.Series(city.index.isin(run_length.index[run_length >= SPELL_SHIFTS]), index=city.index)
sensitivity = [effect(f'freezing (ERA5 temp_min <= {FREEZE_HARD:.0f})',
                      city['era5_temp_min'].le(FREEZE_HARD), '2022-2025', TEST_START),
               effect(f'freezing, spell of {SPELL_SHIFTS}+ shifts', freeze & in_spell, '2022-2025', TEST_START),
               effect('freezing, single shift', freeze & ~in_spell, '2022-2025', TEST_START)]
print('\nFreezing sensitivity (thresholds fixed in advance):')
print(pd.DataFrame(sensitivity).round(3).to_string(index=False))

# which stations carry the strong-wind effect (descriptive)
wind_days = city.index[shift_flags['era5']['flag_wind'].eq(1)]
by_station = df.set_index([DATE_COL, STATION_COL])[TARGET]
extra = {}
for d in wind_days:
    _, window = usual(d)
    if window:
        base = by_station.loc[window].groupby(level=STATION_COL).mean()
        extra[d] = by_station.loc[d] - base
extra = pd.DataFrame(extra).T
print(f'\nExtra calls per strong-wind shift by station, 2022-2025 ({len(extra)} shifts), top and bottom 5:')
station_extra = extra.mean().sort_values(ascending=False).round(2)
print(pd.concat([station_extra.head(5), station_extra.tail(5)]).to_string())

# 4. (c) rolling CV
print('\n=== (c) 2025 rolling CV ===')
X_base, base_numeric = build_features(df, extra_exclude=EXCLUDE + SOURCE_COLS)
# the no-weather model must use exactly the shared baseline (plus station/weekday dummies)
assert sorted(base_numeric) == sorted(BASELINE_NUMERIC), f'unexpected features {sorted(set(base_numeric) ^ set(BASELINE_NUMERIC))}'
assert all(c in base_numeric or c.startswith(('st_', 'wd_')) for c in X_base.columns)


def features(source):
    """Feature matrix with the three flags from 'era5' or 'fc' (or none); checks the columns used."""
    if source is None:
        return X_base
    d = df.join(shift_flags[source], on=DATE_COL)
    X, numeric = build_features(d, extra_exclude=EXCLUDE + SOURCE_COLS)
    assert sorted(numeric) == sorted(BASELINE_NUMERIC + FLAGS), sorted(set(numeric) ^ set(BASELINE_NUMERIC + FLAGS))
    return X


X = {s: features(s) for s in (None, 'era5', 'fc')}
assert X['era5'].columns.equals(X['fc'].columns)
row_era5_any = df[DATE_COL].map(era5_any)
row_fc_wind = df[DATE_COL].map(shift_flags['fc']['flag_wind']).eq(1)


def estimate_r(tr, va, flagged, val_start):
    """Mean calls / usual on flagged training shifts; usual shifts must also be training shifts."""
    allowed = set(tr[DATE_COL].unique())
    days = [d for d in flagged if d in allowed]
    ratios, used = [], set(days)
    for d in days:
        u, window = usual(d, allowed)
        if not np.isnan(u):
            ratios.append(city.loc[d, 'calls'] / u)
            used.update(window)
    # leakage guard: every shift used (flagged and usual) is a training shift, before the gap and the fold
    assert used <= allowed and (not used or max(used) < pd.Timestamp(val_start) - pd.Timedelta(days=2))
    assert not used & set(va[DATE_COL].unique()), 'r uses validation shifts'
    n_events = len(set(events(days))) if days else 0
    return (np.mean(ratios) if ratios else 1.0), len(ratios), n_events


# name: (training flags, validation flags, first training shift, sample weight for extreme shifts)
MODEL_VERSIONS = {
    'A  no weather': (None, None, None, 1),
    'F  flags, ERA5 train / forecast val': ('era5', 'fc', None, 1),
    **{f'W{w} flags + weight {w}': ('era5', 'fc', None, w) for w in WEIGHTS},
    "F' flags, forecast / forecast": ('fc', 'fc', FC_START, 1),
    f"A' no weather, from {FC_START}": (None, None, FC_START, 1),
}
A, A_PRIME = 'A  no weather', f"A' no weather, from {FC_START}"
R_VERSIONS = {'R  A x r (r from forecast-flagged shifts)': shift_flags['fc']['flag_wind'],
              'R5 A x r (r from ERA5 strong-wind shifts)': shift_flags['era5']['flag_wind']}

results, preds, r_rows = [], {n: [] for n in [*MODEL_VERSIONS, *R_VERSIONS]}, []
for start, tr, va in iter_folds(df):
    for name, (train_src, val_src, train_from, w) in MODEL_VERSIONS.items():
        t = tr if train_from is None else tr[tr[DATE_COL] >= train_from]
        weight = np.where(row_era5_any.loc[t.index], w, 1.0) if train_src == 'era5' and w != 1 else None
        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X[train_src].loc[t.index], t[TARGET], sample_weight=weight)
        pred = model.predict(X[val_src].loc[va.index])
        preds[name].append(pd.Series(pred, index=va.index))
        results.append({'fold': start, 'version': name, **metrics(va, pred)})
        if name == A:
            pred_a = pred
    for name, flag in R_VERSIONS.items():
        r, n_shifts, n_events = estimate_r(tr, va, flag.index[flag.eq(1)], start)
        pred = pred_a * np.where(row_fc_wind[va.index], r, 1.0)
        preds[name].append(pd.Series(pred, index=va.index))
        results.append({'fold': start, 'version': name, **metrics(va, pred)})
        r_rows.append({'fold': start, 'version': name, 'r': r, 'training shifts': n_shifts,
                       'training events': n_events, 'validation shifts adjusted': int(row_fc_wind[va.index].groupby(va[DATE_COL]).first().sum())})

results = pd.DataFrame(results)
scores = ['station_MAE', 'station_RMSE', 'citywide_MAE']
print('\nr per fold (estimated on training shifts only, usual shifts inside the training period):')
print(pd.DataFrame(r_rows).round(3).to_string(index=False))
print('\nPer fold:')
print(results.round(3).to_string(index=False))
mean = results.groupby('version', sort=False)[scores].mean()
print("\nMean over the 4 folds (difference from A; compare F' with A'):")
print(mean.join(mean - mean.loc[A], rsuffix='_vs_A').round(3).to_string())
f_prime = "F' flags, forecast / forecast"
print("F' minus A' (same training rows): "
      + ', '.join(f'{s} {mean.loc[f_prime, s] - mean.loc[A_PRIME, s]:+.3f}' for s in scores))

# extreme-weather subsets, 4 folds pooled (all of 2025)
val = df.loc[pd.concat(preds[A]).index]
val_flags = {k: val[DATE_COL].map(v) for k, v in [('era5_wind', shift_flags['era5']['flag_wind']),
                                                   ('era5_freeze', shift_flags['era5']['flag_freeze']),
                                                   ('era5_snow', shift_flags['era5']['flag_snow']),
                                                   ('fc_wind', shift_flags['fc']['flag_wind'])]}
subsets = {f'strong wind, ERA5 gust >= {WIND_ERA5:.0f}': val_flags['era5_wind'].eq(1),
           f'strong-wind forecast, gust >= {WIND_FC:.0f} (what the app sees)': val_flags['fc_wind'].eq(1),
           f'freezing, ERA5 temp_min <= {FREEZE:.0f}': val_flags['era5_freeze'].eq(1),
           f'snow, ERA5 snow_sum >= {SNOW:.0f} cm': val_flags['era5_snow'].eq(1),
           'all 2025': pd.Series(True, index=val.index)}
print('\nSubsets of 2025 (4 folds pooled):')
for label, mask in subsets.items():
    sub = val[mask]
    days = sub[DATE_COL].unique()
    n_events = '' if label == 'all 2025' else f', {len(set(events(days))) if len(days) else 0} events'
    print(f"\n{label}: {len(days)} shifts{n_events}")
    if len(sub):
        # citywide_bias: mean of (predicted - actual) citywide calls; > 0 means over-predicted
        table = pd.DataFrame({n: {**metrics(sub, pd.concat(p).loc[sub.index].to_numpy()),
                                  'citywide_bias': (pd.concat(p).loc[sub.index].groupby(sub[DATE_COL]).sum()
                                                    - sub.groupby(DATE_COL)[TARGET].sum()).mean()}
                              for n, p in preds.items()}).T[['station_MAE', 'citywide_MAE', 'citywide_bias']]
        print(table.round(3).to_string())

# each strong-wind or strong-wind-forecast shift of 2025: citywide actual vs predictions
case = val[val_flags['era5_wind'].eq(1) | val_flags['fc_wind'].eq(1)]
cases = pd.DataFrame({n: pd.concat(p).loc[case.index].groupby(case[DATE_COL]).sum() for n, p in preds.items()})
cases.columns = [c.split()[0] for c in cases.columns]
cases.insert(0, 'actual', case.groupby(DATE_COL)[TARGET].sum())
cases.insert(0, 'fc_gust', city.loc[cases.index, 'fc_gust_max'])
cases.insert(0, 'era5_gust', city.loc[cases.index, 'era5_gust_max'])
print('\nEach 2025 shift with ERA5 strong wind or a strong-wind forecast (citywide calls):')
print(cases.round(1).to_string())

# 5. (d) ERA5 strong-wind shifts 2003-2019
print('\n=== (d) ERA5 strong-wind shifts, 2003-2019 (2020-21 left out) ===')
if os.path.exists(HISTORY_COUNTS):
    history = pd.read_csv(HISTORY_COUNTS)
else:
    stations = pd.read_csv('data/sfd_stations.csv')
    params = {'latitude': stations['lat'].mean(), 'longitude': stations['lon'].mean(),
              'hourly': 'wind_gusts_10m', 'models': 'era5', 'timezone': 'GMT',
              'start_date': '2002-12-31', 'end_date': '2020-01-01'}
    r = requests.get(ARCHIVE_URL, params=params, timeout=600)
    r.raise_for_status()
    h = pd.DataFrame(r.json()['hourly'])
    # same alignment as step 8: gusts cover the hour before the timestamp; local time minus 8 h
    gust = h.set_index(pd.to_datetime(h['time']))['wind_gusts_10m'].shift(-1, freq='h')
    local = gust.index.tz_localize('UTC').tz_convert('America/Los_Angeles').tz_localize(None)
    g = gust.groupby((local - pd.Timedelta(hours=8)).normalize())
    shift_gust = g.max().where(g.count().eq(g.size()))
    shift_gust = shift_gust[(shift_gust.index >= HISTORY_START) & (shift_gust.index <= HISTORY_END)]
    strong = shift_gust.index[shift_gust >= WIND_ERA5]
    ev = events(strong)
    history = pd.DataFrame({'year': range(2003, 2020)})
    history['shifts'] = history['year'].map(shift_gust.notna().groupby(shift_gust.index.year).sum())
    history['strong_wind_shifts'] = history['year'].map(pd.Series(strong.year).value_counts()).fillna(0).astype(int)
    history['strong_wind_events'] = history['year'].map(
        pd.Series(ev.groupby(ev).apply(lambda e: e.index.min().year)).value_counts()).fillna(0).astype(int)
    history.to_csv(HISTORY_COUNTS, index=False)
print(history.to_string(index=False))
recent = shift_flags['era5']['flag_wind'].eq(1)
recent_ev = events(city.index[recent])
print(f"2003-2019: {history['strong_wind_shifts'].sum()} shifts, {history['strong_wind_events'].sum()} events "
      f"({history['strong_wind_shifts'].mean():.1f} shifts/year); 2022-2025: {int(recent.sum())} shifts, "
      f"{recent_ev.nunique()} events ({recent.sum() / 4:.1f} shifts/year)")
