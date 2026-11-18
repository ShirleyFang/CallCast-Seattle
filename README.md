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
| Step 15      | `scripts/step15_weather_extremes.py` | `data/modeling_dataset_weather.csv` → forecast skill, effect size and CV for rare extreme weather (printed) |

Scripts run from the repository root (`uv run python scripts/<name>.py`). Notebooks live in `notebooks/` and read files with `../data/`.
`scripts/cv_utils.py` holds the folds, features and metrics shared by steps 5 and 6.

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
- **Features:** calendar (`weekday`, `month`, `year`), US federal holidays (`is_holiday`, `is_holiday_eve`), and recent calls per station using only shifts at least 3 earlier (`lag3`, `lag7`, `avg7`, `avg28`, `same_weekday_avg4`).
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

Means over the 4 folds of 2025. The 3 features are `precip_sum`, `snow_sum` and `temp_min`. C-cal maps the forecast to the ERA5 scale with quantile mapping, fitted in each fold only on training shifts that have both (2024-01-21 to the training cutoff). An assert checks that no validation shift is used. Fold 1 has only 21 such shifts with snowfall, so its snowfall is left uncalibrated.

- **Main result:** weather that is known when the forecast is made does not improve the model. Every forecast version (C, C-cal, C′, C′3) has a citywide MAE at or above A. At the station level all versions are within 0.04 calls, since the weather is one value for the whole city. Even perfect weather (B, B3) lowers citywide MAE by only 0.4–0.6, and only in 2 of the 4 folds. The differences between folds are larger (17.6–20.8).
- **Forecast vs. actual weather:** training on ERA5 and validating on the forecast (C) is the worst version, because GFS is drier and less windy than ERA5. Calibration removes most of the bias (gusts −4.1 → +0.3 km/h, minimum temperature −1.0 → +0.2 °C on 2025 shifts) and most of the loss (+1.93 → +0.22), but it still does not beat A.
- **C′ vs A′:** with the same training rows (2024-01-21 onward, only 339 shifts in fold 1), the forecast adds +0.47 citywide MAE with all 6 features and −0.06 with 3. So the forecast does not help, and the extra features add noise.
- **Extreme weather** (2025, ERA5 top 10% by rain or gusts, or any snow): perfect weather helps a little (citywide MAE −1.0 on heavy rain, −2.0 on strong wind, −1.2 on snow). On heavy rain and strong wind every forecast version is worse than A. On snow only C′ is lower (20.6), and C doubles the error (21.9 → 43.9; 22.6 after calibration). These subsets have only 11–37 shifts. A and A′, which differ only in training rows, already differ by up to 0.9 on them, so these numbers are noisy.
- **Limits:** one untuned model and one point of weather for the whole city. GFS forecasts exist only from 2024-01-21 (snowfall from 2024-12-09), which leaves C′ and the calibration little training data. The results cover 2025 only, and 2026 is still unused.

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
