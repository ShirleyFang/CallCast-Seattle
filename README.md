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
- **Coverage:** the forecasts start on 2024-01-21 (`fc_snow_sum` on 2024-12-09), so `fc_*` is empty before that. 2025 and 2026 are complete.
- **Shifts:** hours are converted to Seattle time and grouped with the cleaning rule (local time minus 8 hours), so daylight-saving days have 23 or 25 hours. Precipitation, snowfall and gusts cover the hour before their timestamp, so they are shifted to the hour where they start. A shift with any missing hour is left empty.
- **Features per shift:** `precip_sum` (mm), `precip_max_hr` (mm in one hour), `snow_sum` (cm), `temp_max` and `temp_min` (°C), `gust_max` (km/h).

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
