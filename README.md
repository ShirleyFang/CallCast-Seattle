# CallCast-Seattle

Forecasts Seattle Fire Department 911 calls per station per 08:00–08:00 shift, 3 shifts ahead, to support staffing decisions.

## Setup

```bash
uv sync
```

To retrain the model locally:

```bash
uv run python scripts/train_model.py
```

This generates `models/poisson_gbm.joblib` and `model_metadata.json`,
which are required for the Streamlit app's "Trained model" mode.

`scripts/fetch_concerts.py` needs a free [setlist.fm API key](https://www.setlist.fm/settings/api). Copy `.env.example` to `.env` and paste the key there. `.env` is in `.gitignore`, so the key is never committed.

The raw export `data/Seattle_Real_Time_Fire_911_Calls_20261029.csv` (289 MB) is not in git. Download [Seattle Real Time Fire 911 Calls](https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj) as CSV into `data/` before running the cleaning notebook. Everything after cleaning starts from `data/station_shift_counts.csv`, which is in git.

## Pipeline

| Step         | File                                | Input → output                                                                                            |
| ------------ | ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Cleaning     | `notebooks/data_clean.ipynb`        | raw export → `data/station_shift_counts.csv`                                                              |
| Step 1       | `scripts/step1_inspect.py`          | `station_shift_counts.csv` → structure checks (printed)                                                   |
| Step 2       | `scripts/step2_sec2.py`             | `station_shift_counts.csv` → target distribution and histogram (proposal Section 2)                       |
| Step 3       | `scripts/step3_baseline.py`         | `station_shift_counts.csv` → baseline scores (proposal Section 3)                                         |
| Lag analysis | `notebooks/lag_analysis.ipynb`      | `station_shift_counts.csv` → choice of recent-call features                                               |
| Step 4       | `scripts/step4_features.py`         | `station_shift_counts.csv` → `data/modeling_dataset.csv`                                                  |
| Step 5       | `scripts/check_modeling_dataset.py` | `data/modeling_dataset.csv` → sanity checks (printed)                                                     |
| Step 6       | `scripts/evaluate_models.py`        | `data/modeling_dataset.csv` → rolling-CV scores of baselines and first-pass models (printed)              |
| Step 7       | `scripts/train_model.py`            | `data/modeling_dataset.csv` → trained Poisson GBM (`models/poisson_gbm.joblib`) and `model_metadata.json` |
| Prediction   | `predict.py`                        | saved model + feature rows → predicted calls per station                                                  |
| Step 8       | `scripts/step8_weather.py`          | `data/modeling_dataset.csv` + Open-Meteo → `data/modeling_dataset_weather.csv`                            |
| Step 9       | `scripts/step9_weather_compare.py`  | `data/modeling_dataset_weather.csv` → rolling-CV scores with and without weather (printed)                |
| Step 10      | `scripts/evaluate_by_station.py`    | `data/modeling_dataset.csv` → per-station model comparison and `data/station_errors.csv`                  |
| Step 11      | `scripts/evaluate_lag_features.py`  | `data/modeling_dataset.csv` + `station_shift_counts.csv` → rolling-CV scores with extra lag features (printed) |
| Step 12      | `scripts/evaluate_calendar_features.py` | `data/modeling_dataset.csv` → rolling-CV scores with extra calendar features (printed)                |
| Step 13      | `scripts/fetch_events.py`, `scripts/fetch_concerts.py` | public schedules → `data/events.csv` (home games), `data/concerts.csv` (concerts)      |
| Step 14      | `scripts/evaluate_events.py`        | `data/modeling_dataset.csv` + event schedules → event effects and rolling-CV scores (printed)             |
| Step 15      | `scripts/step15_weather_extremes.py` | `data/modeling_dataset_weather.csv` → forecast skill, effect size and CV for rare extreme weather (printed) |

Scripts run from the repository root (`uv run python scripts/<name>.py`). Notebooks live in `notebooks/` and read files with `../data/`.
`scripts/cv_utils.py` holds the folds, features and metrics shared by the evaluation steps (6, 9, 11–14). `scripts/feature_sets.py` holds the baseline feature list and the candidate features for Steps 11–14.

### Cleaning (`notebooks/data_clean.ipynb`)

- Keeps shifts from 2022-01-01 08:00 to 2026-10-29 08:00. 2020–21 were affected by COVID.
- Maps all 168 call types to groups with `data/sfd_type_mapping.csv`, and drops TEST and Fast Back Up (follow-up records for a call already counted).
- Removes Triaged Incident and Nurseline/AMR. Both stop on 2025-05-31 and no other code takes their place; the checks are in the notebook.
- Assigns each call to its nearest station (`data/sfd_stations.csv`) and to its shift (a call at 03:00 belongs to the previous day's shift).
- Flags shifts with a data-feed outage (an hour with no call anywhere in the city) as `Masked`.
- Saves one row per station per shift, with 0 when there was no call: the total `calls`, a count per group, and `Masked`.

### Step 1: Inspect the table (`scripts/step1_inspect.py`)

Checks that `station_shift_counts.csv` has the expected columns, date range, and one row for every station and shift (34 stations × 1,762 shifts). Nothing is changed.

### Step 2: Data reality check (`scripts/step2_sec2.py`)

Drops outage shifts, then reports the target distribution (median, share of zero-call shifts, variance-to-mean ratio), saves a histogram, prints one random row to explain, and compares average calls before and after 2025-05-31. Since Triaged Incident and Nurseline/AMR are now removed in cleaning, that before/after comparison no longer shows their drop.

### Step 3: Baseline (`scripts/step3_baseline.py`)

Splits by year (train 2022–2024, validation 2025, test 2026, outage shifts dropped) and always predicts the training mean, with no inputs. Every model has to beat these test scores:

| Level              | Training mean | Test MAE | Test RMSE |
| ------------------ | ------------- | -------- | --------- |
| Per station-shift  | 9.05          | 5.98     | 9.11      |
| Citywide per shift | 307.8         | 23.4     | 29.7      |

These numbers are after removing the two stopped codes. The proposal reported the earlier version (training mean 10.07, test MAE 6.35, citywide test MAE 37.6).

### Lag analysis (`notebooks/lag_analysis.ipynb`)

Lag plots, autocorrelation and partial autocorrelation of calls per shift, on the training years (2022–2024) only. Lags 1–2 are not known 3 shifts ahead. Among the rest, lag 3 and the weekly lags (7, 14, 21) add new information, and the slowly fading autocorrelation shows a level that changes over weeks. These results set the recent-call features in step 4.

### Step 4: Modeling dataset (`scripts/step4_features.py`)

- **Lead time:** forecasts are made 3 shifts ahead, for example Thursday morning for Saturday. One day is too short to arrange extra staff. Change `MIN_LAG` at the top of the script for a longer lead time.
- **Features:** calendar (`weekday`, `month`, `year`), US federal holidays (`is_holiday`, `is_holiday_eve`), and recent calls per station using only shifts at least 3 earlier (`lag3`, `lag7`, `avg7`, `avg28`, `same_weekday_avg4`). These are the baseline features. Candidate features (special days, events and others; Steps 11–14) are not in this file: `scripts/feature_sets.py` adds them in memory for evaluation.
- **Outages:** outage shifts are left out of the lag calculations, and their rows are dropped afterwards.
- **Missing values:** the lags are empty for the first weeks of 2022 and next to outages. Tree models handle this as is; linear models need those rows dropped.
- **Left out:** the group counts (`fire`, `medical`, ...), since they add up to the target.
- **Forecast rows:** the same features are also built for the 3 shifts after the last known shift, with calls empty, and saved to data/forecast_features.csv for the app. The forecast dates are determined automatically from the latest available shift in the dataset. Past shifts are read from modeling_dataset.csv.
- **Splits:** none in the file. Use rolling cross-validation split by `ShiftDate`
- **Why a script:** cleaning reads the 289 MB raw file, while this step reads the 2 MB table, so features can be changed and rebuilt in seconds with one command.

### Step 5: Check the modeling dataset (`scripts/check_modeling_dataset.py`)

Run this before training. It prints the shape, the missing values per column, and whether a call-group column (which adds up to the target) is still in the file. It reproduces the Step 3 baseline on the modeling dataset (training mean 9.05, test MAE 5.98, citywide test MAE 23.4) and compares each lag column with the calls from that many days earlier. A few mismatches next to outage shifts are expected.

### Step 6: Evaluate models (`scripts/evaluate_models.py`)

Rolling cross-validation with four quarterly folds over 2025: each fold trains on all shifts before the quarter and validates on that quarter. The 2026 test set is not used. It compares three baselines (training mean, station mean, station × weekday mean, lag3) with two first-pass models (Poisson gradient boosting and random forest, no tuning) on station MAE, station RMSE and citywide MAE (the sum of the 34 station forecasts per shift). The `year` column is left out of the features, because in the first fold 2025 is not in the training data and the scores got worse with it. Results are printed, not saved. Each fold trains on all shifts up to two shifts before the quarter (a gap, because the labels of the last two shifts are not yet known when the first shift of the quarter is forecast) and validates on that quarter.

### Step 7: Train and save the model (`scripts/train_model.py`)

Trains a Poisson Gradient Boosting model using shifts from 2022–2025, leaving 2026 out of training for final evaluation. The model uses the same feature configuration as Step 6, excluding `year`.

Run from the repository root:

```bash
uv run python scripts/train_model.py
```

This generates:

- `models/poisson_gbm.joblib`: the trained model, feature column order, preprocessing information, and station × weekday averages.
- `model_metadata.json`: the model name, training cutoff, raw feature columns, and missing-value support required by the Streamlit app.

The model currently uses the first-pass configuration without hyperparameter tuning or weather features. Final model selection and evaluation on the 2026 test set are still pending.

### Prediction interface (`predict.py`)

The prediction interface loads the saved model and applies the same feature encoding used during training.

```python
from predict import predict_shift

predictions = predict_shift(feature_rows)
```

`feature_rows` is a DataFrame containing the required model inputs, including station identifiers. The function returns a DataFrame with `Station` and `predicted_calls`.

The interface has been tested with all 34 stations and successfully integrated with the Streamlit app for historical replay. Future forecasts require the app to load the corresponding rows from `data/forecast_features.csv`.

### Step 8: Weather features (`scripts/step8_weather.py`)

- **Source:** [Open-Meteo](https://open-meteo.com), licensed CC BY 4.0, at one point for the whole city (the mean of the station locations). The hourly data is cached in `data/weather_hourly_era5.csv` and `data/weather_hourly_gfs_global_day3.csv`. Both files are in git, so the script runs without downloading; delete a file to download it again.
- **Two versions:** `era5_*` is the weather that actually happened, from the ERA5 reanalysis (a 0.25° grid model, not weather-station observations). It is the upper bound of a perfect forecast. The nearest ERA5 grid point (47.50, −122.25) is southeast of downtown. Averaging the 4 grid points around the city did not bring ERA5 closer to the forecast (mean rain gap 0.90 → 0.82 mm per shift, temperature error larger), so the single point is kept. ERA5 is still about 0.9 mm per shift wetter than GFS on average. `fc_*` is the GFS forecast (`gfs_global`) available when the prediction is made: `previous_day3` from the Previous Runs API, made at least 72 hours before each hour of the target shift. It follows `MIN_LAG`.
- **Coverage:** the forecasts start on 2024-01-21, so `fc_*` is empty before that. For `fc_snow_sum`, snowfall data starts 2024-12-08 18:00 UTC, so the first complete shift is 2024-12-09. 2025 and 2026 are complete.
- **Shifts:** hours are converted to Seattle time and grouped with the cleaning rule (local time minus 8 hours), so daylight-saving days have 23 or 25 hours. Precipitation, snowfall and gusts cover the hour before their timestamp, so they are shifted to the hour where they start. A shift with any missing hour is left empty.
- **Features per shift:** `precip_sum` (mm), `precip_max_hr` (mm in one hour), `snow_sum` (cm), `temp_max` and `temp_min` (°C), `gust_max` (km/h).

### Step 9: Weather comparison (`scripts/step9_weather_compare.py`)

Runs the same model as Step 6 (Poisson GBM, no tuning, `year` left out) with the folds, training gap and metrics from `cv_utils.py`, with and without the Step 8 weather. 2026 is dropped after loading. Version A reproduces the Poisson GBM scores of Step 6 exactly. A check confirms each version uses only the weather columns it should.

| Version | Training weather                      | Validation weather      | Station MAE | Station RMSE | Citywide MAE (vs A) |
| ------- | ------------------------------------- | ----------------------- | ----------- | ------------ | ------------------- |
| A       | none                                  | none                    | 2.409       | 3.314        | 18.95               |
| B       | ERA5                                  | ERA5                    | 2.414       | 3.318        | 18.58 (−0.37)       |
| B3      | ERA5, 3 features                      | ERA5, 3 features        | 2.406       | 3.309        | 18.38 (−0.57)       |
| C       | ERA5                                  | forecast                | 2.444       | 3.364        | 20.87 (+1.93)       |
| C-cal   | ERA5                                  | forecast mapped to ERA5 | 2.415       | 3.316        | 19.16 (+0.22)       |
| C′      | forecast, from 2024-01-21             | forecast                | 2.427       | 3.359        | 19.65 (+0.71)       |
| C′3     | forecast, from 2024-01-21, 3 features | forecast, 3 features    | 2.418       | 3.335        | 19.12 (+0.18)       |
| A′      | none, from 2024-01-21                 | none                    | 2.417       | 3.326        | 19.18 (+0.23)       |

Means over the 4 folds of 2025, on the baseline features (adding the proposed special days, A would be 18.73). The 3 features are `precip_sum`, `snow_sum` and `temp_min`. C-cal maps the forecast to the ERA5 scale with quantile mapping, fitted in each fold only on training shifts that have both (2024-01-21 to the training cutoff). An assert checks that no validation shift is used. Fold 1 has only 21 such shifts with snowfall, so its snowfall is left uncalibrated.

- **Main result:** weather that is known when the forecast is made does not improve the model. Every forecast version (C, C-cal, C′, C′3) has a citywide MAE at or above A. At the station level all versions are within 0.04 calls, since the weather is one value for the whole city. Even perfect weather (B, B3) lowers citywide MAE by only 0.4–0.6, and only in 2 of the 4 folds. The differences between folds are larger (17.6–20.8).
- **Forecast vs. actual weather:** training on ERA5 and validating on the forecast (C) is the worst version, because GFS is drier and less windy than ERA5. Calibration removes most of the bias (gusts −4.1 → +0.3 km/h, minimum temperature −1.0 → +0.2 °C on 2025 shifts) and most of the loss (+1.93 → +0.22), but it still does not beat A.
- **C′ vs A′:** with the same training rows (2024-01-21 onward, only 339 shifts in fold 1), the forecast adds +0.47 citywide MAE with all 6 features and −0.06 with 3. So the forecast does not help, and the extra features add noise.
- **Extreme weather** (2025, ERA5 top 10% by rain or gusts, or any snow): perfect weather helps a little (citywide MAE −1.0 on heavy rain, −2.0 on strong wind, −1.2 on snow). On heavy rain and strong wind every forecast version is worse than A. On snow only C′ is lower (20.6), and C doubles the error (21.9 → 43.9; 22.6 after calibration). These subsets have only 11–37 shifts. A and A′, which differ only in training rows, already differ by up to 0.9 on them, so these numbers are noisy.
- **Limits:** one untuned model and one point of weather for the whole city. GFS forecasts exist only from 2024-01-21 (snowfall from 2024-12-09), which leaves C′ and the calibration little training data. The results cover 2025 only, and 2026 is still unused.

Steps 10–14 use the same folds, training gap and untuned Poisson GBM as Step 6, and never use 2026. Each script ends with its results in a comment. Steps 11, 12 and 14 also share:

- **Baseline model:** `scripts/feature_sets.py` lists its numeric features explicitly in `BASELINE_NUMERIC` (the Step 6 GBM's inputs, the same as `modeling_dataset.csv`; the model also gets the station and weekday dummies). Columns added to `modeling_dataset.csv` later do not change it.
- **Candidate features:** `feature_sets.py` adds them to the modeling table in memory (`add_special_days`, `add_calendar_candidates`, `add_lag_candidates`, `add_event_candidates`, or `add_candidates` for all), so no extra CSV is saved and `modeling_dataset.csv` keeps only the baseline features. Which candidates enter the final model is decided in the model evaluation.
- **Scoring:** `cv_utils.metrics()` on each fold, then the mean over the folds, as in Step 6.
- **Selection rule:** a feature is kept only if citywide MAE is lower in at least 3 of the 4 folds and station MAE is not worse. With only 4 folds this is a first screen: a feature with no effect still passes about 31% of the time (5 of 16 outcomes), so features that affect only a few days are also checked on those days.

Step 11 compares with the baseline model alone (citywide MAE 18.95). Steps 12 and 14 test the other candidates on top of the baseline plus the special days, the candidates with the clearest evidence.

### Step 10: Per-station models (`scripts/evaluate_by_station.py`)

Does a different model work better at different stations? Compares baselines, the global GBM and random forest (one model for all stations), and a per-station GBM and Poisson GLM (one model per station, trained on its ~1,000 rows).

| Model                  | Station MAE | Citywide MAE |
| ---------------------- | ----------- | ------------ |
| Global GBM             | 2.409       | 18.95        |
| Per-station GLM        | 2.431       | 19.98        |
| Station × weekday mean | 2.448       | 20.71        |
| Global random forest   | 2.457       | 20.03        |
| Per-station GBM        | 2.718       | 21.40        |

- The best model beats the global GBM by more than 2% at only 3 of 34 stations. Picking a model per station on folds 1–3 did not hold up on fold 4, so the global GBM is kept everywhere.
- The stations farthest from the Poisson noise floor (the error left even by a perfect forecast of the mean) are SFD 10 (+17%), SFD 9 (+16%) and SFD 17 (+13%).
- `data/station_errors.csv` saves one row per station (MAE per model, noise floor, best model, address, lat/lon) for the app.

### Step 11: Extra lag features (`scripts/evaluate_lag_features.py`)

Adds history features built from the call data, each using only shifts at least 3 earlier: citywide level (`citywide_ratio`, `citywide_lag3_ratio`), the 3 nearest stations (`neighbor_lag3`), trend (`avg7 / avg28`), volatility (`std28`), calls by type 3 shifts earlier, and the same weekday one year earlier (`lag364`).

**Result:** none helps. Station MAE changes by less than 0.3%, and citywide MAE is 1–3% worse with every group. The existing lags already hold this information.

### Step 12: Calendar features (`scripts/evaluate_calendar_features.py`)

| Version                                   | Station MAE | Citywide MAE | Folds better |
| ----------------------------------------- | ----------- | ------------ | ------------ |
| Current GBM                               | 2.409       | 18.95        | –            |
| + special days                            | 2.406       | 18.73        | see below    |
| + special days + month start (days 1–3)   | 2.407       | 18.83        | 1 of 4       |
| + special days + UW quarter in session    | 2.410       | 18.94        | 1 of 4       |
| + special days + long weekend             | 2.411       | 19.11        | 1 of 4       |
| + special days + day length               | 2.410       | 19.03        | 1 of 4       |
| + special days + Seattle Public Schools break | 2.404   | 18.78        | 2 of 4       |

"Folds better" counts the folds where citywide MAE is lower than "+ special days".

- **Special days** (`is_nye`, `is_day_after_thanksgiving`, `is_xmas` for Dec 24–26) are proposed for the final model; they are not in `modeling_dataset.csv` (the baseline), and `feature_sets.add_special_days` builds them. All of them fall in Q4, so the fold count says nothing about them: the Q1–Q3 changes are noise. On the 5 special-day shifts of 2025, citywide MAE drops from 44.0 to 26.3 and every one of them improves; the other shifts are unchanged (18.60 → 18.63). Over 2004–2025, calls were higher on New Year's Eve in 22 of 22 years and lower on the day after Thanksgiving in 20 of 22; the existing holiday flags miss both.
- **Nothing else** adds anything on top. Month and weekday already cover day length and long weekends. The UW calendar does not help even at SFD 17 and SFD 22 next to campus. On school-break weekdays, citywide calls are about the same as on school days (293.6 vs 296.8 per shift, Feb and Apr 2022–2024).
- UW quarter dates come from the UW academic calendars. SPS breaks follow fixed rules in the district calendars: mid-winter break is the week of Presidents Day, spring break the week starting the 2nd Monday of April.

### Step 13: Event schedules (`scripts/fetch_events.py`, `scripts/fetch_concerts.py`)

- **Home games**, 2022–2026 (`data/events.csv`): Seahawks, Husky football, Mariners, Sounders, Kraken and Storm, from the MLB, NHL and ESPN public APIs. Only games at Lumen Field, T-Mobile Park, Climate Pledge Arena and Husky Stadium are kept. The shift is the 08:00–08:00 shift the game starts in.
- **Concerts**, 2022–2026 (`data/concerts.csv`): from setlist.fm (needs the API key, see Setup), at the same venues. One row per venue and night. Nights with a home game at the same venue are dropped (anthem and halftime shows). `size` is `stadium` (Lumen Field, T-Mobile Park) or `arena` (Climate Pledge Arena).
- Schedules are published weeks to months ahead, so they are known when a forecast is made. setlist.fm lists only shows that took place; for large venues, cancellations are rare.

### Step 14: Event effects (`scripts/evaluate_events.py`)

- **City-wide event flags** (one column per event kind, the same at every station) make citywide MAE worse in all 4 folds (18.73 → 18.92). Events change calls only next to the venue.
- **Per station:** on 2022–2024 only, three event-station pairs show a clear effect (at least 10 events, +10% or more, t ≥ 2): Husky football at SFD 22 (+4.4 calls, +111%), Seahawks at SFD 10 (+3.8, +11%) and stadium concerts at Lumen Field at SFD 10 (+4.1, +11%). Mariners, Sounders, Kraken, Storm and arena concerts add 0–4%.
- **Post-model adjustment:** adding each pair's average extra calls (learned from the training shifts) to the GBM forecast cuts the error on those 19 shifts in 2025 from 7.13 to 6.17 (−13%), and the bias from −4.8 to −0.8 calls. Overall MAE is unchanged (18.75). Using the pairs as a GBM feature instead helps less (6.75).
- **Uplift vs residual:** the amount above is the raw uplift (event shifts vs similar shifts). The recommended amount is the GBM's own mean out-of-fold residual on the pair's training event shifts, which does not count effects the model already predicts. Both give the same MAE on the 19 shifts (6.17). Per pair, the uplift over-adjusts at Husky football (bias +2.07), and the residual adds about 0.6 fewer calls in every fold (+1.56). At SFD 10 both still under-forecast. With 4–8 shifts per pair in 2025, the difference is within noise.
- City permits for festivals and parades (Seattle Open Data `dm95-f8w5`) were also checked. There is no concert data, 2025 is almost empty, and large permitted events show no citywide error in 2023–2024. They are not used.

[docs/POST_MODEL_ADJUSTMENT.md](docs/POST_MODEL_ADJUSTMENT.md) describes the adjustment in full: how the pairs were chosen, how the extra calls are computed, where the extra calls happen (within 0.5 km of the venue), how it could fit into `train_model.py` and `predict.py`, and its limits. It is a proposal: the model and the app do not use it yet.

### Step 15: Rare extreme weather (`scripts/step15_weather_extremes.py`)

Step 9 found that weather matters only on rare extreme shifts. This step looks at them directly: strong wind (ERA5 `gust_max` ≥ 70 km/h), freezing (`temp_min` ≤ 0 °C) and snow (`snow_sum` ≥ 1 cm). It uses the same folds, training gap, metrics and untuned Poisson GBM as Steps 6 and 9, on the baseline features (A = 18.95), and drops 2026. All thresholds were fixed before looking at results. The forecast strong-wind threshold is 65 km/h, because GFS gusts are about 4 km/h weaker than ERA5 (Step 8); the hit/miss counts below were not used to choose it. Back-to-back extreme shifts count as one event.

- **(a) Can the forecast see it?** From 2024-01-21 to 2025 (706 shifts): a forecast gust ≥ 65 catches 6 of the 10 ERA5 strong-wind shifts, and 8 of its 14 warnings are false alarms. Freezing is easy (19 of 20 caught, 13 false alarms). Snow ≥ 1 cm is never forecast: 0 of 4 caught.
- **(b) Effect size:** calls compared with usual (same weekday ±4 weeks, non-extreme shifts), 2022–2025, with a 95% interval from resampling whole events. Strong wind: +12.2% (+7.5% to +18.5%; 17 shifts, 15 events). Freezing: +11.4% (+1.6% to +24.8%; 54 shifts but only 18 events). Snow: +3.8% (−0.9% to +8.9%; 20 shifts, 15 events), no clear effect. Some strong-wind shifts fall near holidays (2025-10-31 Halloween, 2022-12-27 after Christmas), so part of the wind effect may be a holiday effect. The extra calls are mostly at SFD 2, 39 and 24.
- **(c) Ways to use it**, 2025 rolling CV (means over the 4 folds; the last two columns are citywide MAE on 2025 subsets):

| Version | What it does                                                            | Station MAE | Citywide MAE (vs A) | Strong wind, ERA5 (7 shifts) | Strong-wind forecast (9 shifts) |
| ------- | ----------------------------------------------------------------------- | ----------- | ------------------- | ---------------------------- | ------------------------------- |
| A       | no weather                                                              | 2.409       | 18.95               | 26.7                         | 21.1                            |
| F       | 0/1 flags: ERA5 in training, forecast in validation                     | 2.442       | 20.70 (+1.76)       | 24.6                         | 28.9                            |
| W5      | F, extreme training shifts weighted 5×                                  | 2.454       | 20.95 (+2.01)       | 24.2                         | 33.3                            |
| W20     | F, extreme training shifts weighted 20×                                 | 2.485       | 21.68 (+2.73)       | 24.2                         | 34.8                            |
| F′      | forecast flags in training and validation, from 2024-01-21              | 2.420       | 19.67 (+0.73)       | 34.5                         | 34.7                            |
| A′      | no weather, from 2024-01-21                                             | 2.417       | 19.18 (+0.23)       | 26.1                         | 21.8                            |
| R       | A × r on strong-wind forecasts, r from forecast-flagged training shifts | 2.410       | 18.99 (+0.05)       | 26.0                         | 23.2                            |
| R5      | A × r, r from ERA5 strong-wind training shifts                          | 2.415       | 19.44 (+0.50)       | 32.6                         | 41.4                            |

R scales the no-weather prediction when the forecast says strong wind. In each fold, r is estimated only from training shifts: the flagged shifts and their ±4-week usual shifts all fall before the training cutoff, and an assert checks that no validation shift is used. R's r is 1.03–1.05, because forecast-flagged training shifts include false alarms; R5's r is 1.13–1.14, the effect of real storms.

- **Main result:** no method beats A. R comes closest (+0.05): it helps a little on real storms (26.7 → 26.0) but loses on false alarms (21.1 → 23.2). Applying the full storm effect (R5) to forecasts that are wrong half the time over-predicts by 33 calls on the flagged shifts. F′ is +0.50 against A′, which uses the same training rows.
- **Why the flags fail:** most of the damage comes from the freezing flag. On 2025's freezing shifts, F over-predicts by 39 calls citywide (MAE 24.3 → 54.2). The ERA5 freezing shifts in training include the late-December 2022 cold spell, which coincided with the Christmas peak (646 calls on 2022-12-23), so the model most likely learned that freezing means far more calls. Weighting the extreme shifts makes it worse.
- **Sample size:** 2025 has only 7 strong-wind shifts (5 events), 6 of them in fold 4. Folds 2 and 3 have no strong-wind forecast, so R equals A there. Freezing has 19 shifts but 4 events, snow 4 shifts. These are case studies, not statistics; the script prints each 2025 strong-wind shift with the actual and predicted citywide calls.
- **(d) More history:** ERA5 has 106 strong-wind shifts (82 events) in 2003–2019, 6.2 per year against 4.2 in 2022–2025, so about 6 times more events (2020–21 left out). This is not suitable for retraining: call coding changed several times between 2014 and 2022, so older counts are not on the same scale as 2022 onward. The older years could firm up the effect estimate, since a ratio to the surrounding weeks is less sensitive to slow coding changes, but that needs the raw export and would not fix the forecast's false alarms. Only the yearly counts are saved (`data/era5_strong_wind_counts_2003_2019.csv`).
- **Recommendation:** keep weather out of the model. In the app, a strong-wind forecast can at most show a note marked for reference only (for example, past storms averaged about 12% more calls, and about half of such forecasts were false alarms), without changing the predicted numbers.

## Streamlit app

```bash
uv sync --frozen
uv run streamlit run app.py
```

The app only needs the CSVs already in `data/`, not the raw 911 export. Pick a shift date and a station to see:

- expected calls for each station and the citywide total
- a station map and a ranked table you can download as CSV
- that station's past calls, weekday and monthly averages, and call types

"Historical reference" uses each station's average for the same weekday in earlier shifts.

"Trained model" uses the saved Poisson GBM through `predict.py`. The model is loaded from `models/poisson_gbm.joblib`, with its configuration and training cutoff defined in `model_metadata.json`.

The trained model has been integrated and tested with the Streamlit app. The current app supports historical replay for eligible dates after the model's training cutoff. Future forecasting and weather inputs are not yet integrated.

The hourly breakdown requires `data/hourly_profile.csv`.

The app plans the way a supervisor would: on Thursday at 08:00 for Saturday's shift. So it only uses shifts that had finished by then (Wednesday and earlier).

Tests:

```bash
uv run python -m unittest discover -s tests -v
```

[docs/APP_WALKTHROUGH.md](docs/APP_WALKTHROUGH.md) goes through the app screen by screen.
