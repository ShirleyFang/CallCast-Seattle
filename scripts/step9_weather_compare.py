# What it does
# Compares the same model with and without the weather features from scripts/step8_weather.py,
# using Elisa's rolling CV over 2025 (folds, training gap and metrics from cv_utils.py).
#   A      no weather
#   B      ERA5 weather in training and validation (perfect-forecast upper bound)
#   C      ERA5 in training, GFS forecast in validation (what the model would see in use)
#   C-cal  as C, but the forecast is first mapped to the ERA5 scale (quantile mapping fitted
#          per fold on training shifts that have both, i.e. 2024-01-21 to the training cutoff)
#   C'     GFS forecast in training and validation; training starts 2024-01-21 (first forecast shift)
#   A'     no weather, same training rows as C' (separates "less data" from "forecast weather")
#   B3, C'3  as B and C', with only 3 weather features (SMALL)
# Also reports errors on 2025 shifts with extreme weather. 2026 is the test set and is dropped.
# Results are printed, not saved.
#
# Run it from the repository root: uv run python scripts/step9_weather_compare.py

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from cv_utils import DATE_COL, TARGET, build_features, iter_folds, load_dataset, metrics


# constant: model and data (swap here if the shared setup changes)
MODEL_PARAMS = {'loss': 'poisson', 'max_iter': 200, 'random_state': 0}
DATA_PATH = 'data/modeling_dataset_weather.csv'
TEST_START = '2026-01-01'          # 2026 is never used
FC_START = '2024-01-21'            # first complete shift with GFS forecasts
EXCLUDE = ['year']                 # same as evaluate_models.py: year is unseen in fold 1
EXTREME_QUANTILE = 0.9             # "extreme" = top 10% of 2025 shifts by ERA5 value
CAL_QUANTILES = np.linspace(0, 1, 101)
MIN_CAL_SHIFTS = 60                # fewer shifts with both values: keep the raw forecast

WEATHER = ['precip_sum', 'precip_max_hr', 'snow_sum', 'temp_max', 'temp_min', 'gust_max']
SMALL = ['precip_sum', 'snow_sum', 'temp_min']
SOURCE_COLS = [f'{s}_{w}' for s in ('era5', 'fc') for w in WEATHER]
BASE_NUMERIC = ['month', 'is_holiday', 'is_holiday_eve', 'lag3', 'lag7', 'avg7', 'avg28', 'same_weekday_avg4']

# name: (training weather, validation weather, first training shift, weather features)
VERSIONS = {
    'A  no weather': (None, None, None, []),
    'B  ERA5 / ERA5': ('era5', 'era5', None, WEATHER),
    'B3 ERA5 / ERA5, 3 features': ('era5', 'era5', None, SMALL),
    'C  ERA5 / forecast': ('era5', 'fc', None, WEATHER),
    'C-cal ERA5 / calibrated forecast': ('era5', 'fc_cal', None, WEATHER),
    "C' forecast / forecast": ('fc', 'fc', FC_START, WEATHER),
    "C'3 forecast / forecast, 3 features": ('fc', 'fc', FC_START, SMALL),
    "A' no weather, from 2024-01-21": (None, None, FC_START, []),
}
A, A_PRIME = 'A  no weather', "A' no weather, from 2024-01-21"

# 1. read the dataset and drop 2026
df = load_dataset(DATA_PATH)
df = df[df[DATE_COL] < TEST_START]
assert df[DATE_COL].max() < pd.Timestamp(TEST_START)

# weather per row under shared names, from ERA5 or the forecast
weather = {s: df[[f'{s}_{w}' for w in WEATHER]].set_axis(WEATHER, axis=1) for s in ('era5', 'fc')}


def features(wx, keep):
    """Feature matrix with the `keep` weather columns from `wx` (or none); checks the columns used."""
    d = df.copy()
    for w in keep:
        d[f'wx_{w}'] = wx[w]
    # the raw era5_*/fc_* columns are always excluded; only the wx_* copies may enter
    X, numeric = build_features(d, extra_exclude=EXCLUDE + SOURCE_COLS)
    expected = BASE_NUMERIC + [f'wx_{w}' for w in keep]
    assert sorted(numeric) == sorted(expected), f'unexpected features {sorted(set(numeric) ^ set(expected))}'
    assert all(c in numeric or c.startswith(('st_', 'wd_')) for c in X.columns)
    return X


def calibrate(cal_rows, val_start):
    """Quantile-map the forecast to the ERA5 scale, fitted on cal_rows only; returns (weather, n per variable)."""
    assert df.loc[cal_rows, DATE_COL].max() < pd.Timestamp(val_start), 'calibration uses validation shifts'
    shifts = df.loc[cal_rows].drop_duplicates(DATE_COL).index       # one value per shift, not per station
    out, n = weather['fc'].copy(), {}
    for w in WEATHER:
        pair = pd.DataFrame({'fc': weather['fc'].loc[shifts, w], 'era5': weather['era5'].loc[shifts, w]}).dropna()
        n[w] = len(pair)
        if n[w] < MIN_CAL_SHIFTS:
            continue
        # tied forecast quantiles (e.g. many dry shifts) map to the mean of their ERA5 quantiles
        table = pd.Series(np.quantile(pair['era5'], CAL_QUANTILES)).groupby(np.quantile(pair['fc'], CAL_QUANTILES)).mean()
        raw = weather['fc'][w]
        out[w] = pd.Series(np.interp(raw, table.index, table.to_numpy()), index=raw.index).where(raw.notna())
    return out, n


X_cache = {}


def X_for(source, keep):
    key = (source, tuple(keep))
    if key not in X_cache:
        X_cache[key] = features(weather[source] if source else None, keep)
    return X_cache[key]


print(f"{len(df)} rows, {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
print('numeric features without weather:', BASE_NUMERIC)
print('weather features (all):', WEATHER, '| small set:', SMALL)

# forecasts must be complete wherever C' and the validation folds use them (snowfall may be empty)
fc_rows = df[DATE_COL] >= FC_START
assert df.loc[fc_rows, [f'fc_{w}' for w in WEATHER if w != 'snow_sum']].notna().all().all()

# 2. rolling CV: same folds and training gap as evaluate_models.py
rows, preds, cal_check = [], {name: [] for name in VERSIONS}, []
for start, tr, va in iter_folds(df):
    # calibration: training shifts that have both ERA5 and the forecast, nothing from validation
    cal_rows = tr.index[tr[DATE_COL] >= FC_START]
    assert cal_rows.isin(tr.index).all() and not cal_rows.isin(va.index).any()
    fc_cal, cal_n = calibrate(cal_rows, start)
    print(f"\nfold {start}: calibration on {tr.loc[cal_rows, DATE_COL].min().date()} to "
          f"{tr.loc[cal_rows, DATE_COL].max().date()}, shifts per variable: {cal_n}")
    va_shifts = va.drop_duplicates(DATE_COL).index
    cal_check.append(pd.DataFrame({f'{k}_{w}': src.loc[va_shifts, w] - weather['era5'].loc[va_shifts, w]
                                   for k, src in (('raw', weather['fc']), ('cal', fc_cal)) for w in WEATHER}))

    for name, (train_src, val_src, train_from, keep) in VERSIONS.items():
        t = tr if train_from is None else tr[tr[DATE_COL] >= train_from]
        X_val = features(fc_cal, keep) if val_src == 'fc_cal' else X_for(val_src, keep)
        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X_for(train_src, keep).loc[t.index], t[TARGET])
        pred = model.predict(X_val.loc[va.index])
        preds[name].append(pd.Series(pred, index=va.index))
        snow_known = t['fc_snow_sum'].notna().mean() if train_src == 'fc' else np.nan
        rows.append({'fold': start, 'version': name, **metrics(va, pred),
                     'train_from': t[DATE_COL].min().date(), 'train_to': t[DATE_COL].max().date(),
                     'train_shifts': t[DATE_COL].nunique(), 'fc_snow_known': snow_known})

# calibration check on 2025 (reporting only; the mapping never saw these shifts)
cal = pd.concat(cal_check)
print('\nForecast minus ERA5 on 2025 shifts, raw vs calibrated (mean = bias, MAE):')
print(pd.DataFrame({w: {'raw bias': cal[f'raw_{w}'].mean(), 'cal bias': cal[f'cal_{w}'].mean(),
                        'raw MAE': cal[f'raw_{w}'].abs().mean(), 'cal MAE': cal[f'cal_{w}'].abs().mean()}
                    for w in WEATHER}).round(3).to_string())

results = pd.DataFrame(rows)
scores = ['station_MAE', 'station_RMSE', 'citywide_MAE']
pd.set_option('display.width', 200)
print('\nPer fold:')
print(results.round(3).to_string(index=False))

mean = results.groupby('version', sort=False)[scores].mean()
print("\nMean over the 4 folds (difference from A; compare C' versions with A'):")
print(mean.join(mean - mean.loc[A], rsuffix='_vs_A').round(3).to_string())
for name in ["C' forecast / forecast", "C'3 forecast / forecast, 3 features"]:
    print(f"{name} minus A' (same training rows): "
          + ', '.join(f'{s} {mean.loc[name, s] - mean.loc[A_PRIME, s]:+.3f}' for s in scores))

fold1 = results[(results['version'] == "C' forecast / forecast") & (results['fold'] == results['fold'].min())].iloc[0]
print(f"\nNote: in fold 1 C', C'3 and A' train only on {fold1['train_from']} to {fold1['train_to']} "
      f"({fold1['train_shifts']} shifts), and fc_snow_sum is known for {fold1['fc_snow_known']:.0%} "
      f"of those rows (forecast snowfall starts 2024-12-09). Missing snowfall is left empty; "
      f"HistGradientBoosting handles it. The same short window limits the C-cal calibration in fold 1.")

# 3. extreme weather: pool the 4 validation quarters (all of 2025), subsets set by ERA5
val = df.loc[pd.concat(preds[A]).index]
per_shift = val.drop_duplicates(DATE_COL).set_index(DATE_COL)
rain = per_shift['era5_precip_sum'].quantile(EXTREME_QUANTILE)
gust = per_shift['era5_gust_max'].quantile(EXTREME_QUANTILE)
subsets = {
    f'heavy rain (ERA5 precip_sum >= {rain:.1f} mm)': val['era5_precip_sum'] >= rain,
    f'strong wind (ERA5 gust_max >= {gust:.0f} km/h)': val['era5_gust_max'] >= gust,
    'any snow (ERA5 snow_sum > 0)': val['era5_snow_sum'] > 0,
    'all 2025': pd.Series(True, index=val.index),
}
print('\nExtreme-weather shifts in 2025 (4 folds pooled):')
for label, mask in subsets.items():
    sub = val[mask]
    print(f"\n{label}: {sub[DATE_COL].nunique()} shifts")
    table = pd.DataFrame({name: metrics(sub, pd.concat(p).loc[sub.index].to_numpy())
                          for name, p in preds.items()}).T[['station_MAE', 'citywide_MAE']]
    print(table.round(3).to_string())
