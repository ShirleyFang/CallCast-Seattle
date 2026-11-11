# CallCast-Seattle

Forecasts Seattle Fire Department 911 calls per station per 08:00–08:00 shift, 3 shifts ahead, to support staffing decisions.

## Setup

```bash
uv sync
```

The raw export `data/Seattle_Real_Time_Fire_911_Calls_20261029.csv` (289 MB) is not in git. Download [Seattle Real Time Fire 911 Calls](https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj) as CSV into `data/` before running the cleaning notebook. Everything after cleaning starts from `data/station_shift_counts.csv`, which is in git.

## Pipeline

| Step | File | Input → output |
|---|---|---|
| Cleaning | `notebooks/data_clean.ipynb` | raw export → `data/station_shift_counts.csv` |
| Step 1 | `scripts/step1_inspect.py` | `station_shift_counts.csv` → structure checks (printed) |
| Step 2 | `scripts/step2_sec2.py` | `station_shift_counts.csv` → target distribution and histogram (proposal Section 2) |
| Step 3 | `scripts/step3_baseline.py` | `station_shift_counts.csv` → baseline scores (proposal Section 3) |
| Lag analysis | `notebooks/lag_analysis.ipynb` | `station_shift_counts.csv` → choice of recent-call features |
| Step 4 | `scripts/step4_features.py` | `station_shift_counts.csv` → `data/modeling_dataset.csv` |
| Step 5 | `scripts/step5_weather.py` | `modeling_dataset.csv` + Open-Meteo → `data/modeling_dataset_weather.csv` |

Scripts run from the repository root (`uv run python scripts/<name>.py`). Notebooks live in `notebooks/` and read files with `../data/`.

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

| Level | Training mean | Test MAE | Test RMSE |
|---|---|---|---|
| Per station-shift | 9.05 | 5.98 | 9.11 |
| Citywide per shift | 307.8 | 23.4 | 29.7 |

These numbers are after removing the two stopped codes. The proposal reported the earlier version (training mean 10.07, test MAE 6.35, citywide test MAE 37.6).

### Lag analysis (`notebooks/lag_analysis.ipynb`)

Lag plots, autocorrelation and partial autocorrelation of calls per shift, on the training years (2022–2024) only. Lags 1–2 are not known 3 shifts ahead. Among the rest, lag 3 and the weekly lags (7, 14, 21) add new information, and the slowly fading autocorrelation shows a level that changes over weeks. These results set the recent-call features in step 4.

### Step 4: Modeling dataset (`scripts/step4_features.py`)

- **Lead time:** forecasts are made 3 shifts ahead, for example Thursday morning for Saturday. One day is too short to arrange extra staff. Change `MIN_LAG` at the top of the script for a longer lead time.
- **Features:** calendar (`weekday`, `month`, `year`), US federal holidays (`is_holiday`, `is_holiday_eve`), and recent calls per station using only shifts at least 3 earlier (`lag3`, `lag7`, `avg7`, `avg28`, `same_weekday_avg4`).
- **Outages:** outage shifts are left out of the lag calculations, and their rows are dropped afterwards.
- **Missing values:** the lags are empty for the first weeks of 2022 and next to outages. Tree models handle this as is; linear models need those rows dropped.
- **Left out:** the group counts (`fire`, `medical`, ...), since they add up to the target.
- **Splits:** none in the file. Use rolling cross-validation split by `ShiftDate`
- **Why a script:** cleaning reads the 289 MB raw file, while this step reads the 2 MB table, so features can be changed and rebuilt in seconds with one command.

### Step 5: Weather features (`scripts/step5_weather.py`)

- **Source:** [Open-Meteo](https://open-meteo.com), licensed CC BY 4.0, at one point for the whole city (the mean of the station locations). The hourly data is cached in `data/weather_hourly_era5.csv` and `data/weather_hourly_gfs_global_day3.csv`. Both files are in git, so the script runs without downloading; delete a file to download it again.
- **Two versions:** `era5_*` is the weather that actually happened, from the ERA5 reanalysis (a 0.25° grid model, not weather-station observations). It is the upper bound of a perfect forecast. The nearest ERA5 grid point (47.50, −122.25) is southeast of downtown. Averaging the 4 grid points around the city did not bring ERA5 closer to the forecast (mean rain gap 0.90 → 0.82 mm per shift, temperature error larger), so the single point is kept. ERA5 is still about 0.9 mm per shift wetter than GFS on average. `fc_*` is the GFS forecast (`gfs_global`) available when the prediction is made: `previous_day3` from the Previous Runs API, made at least 72 hours before each hour of the target shift. It follows `MIN_LAG`.
- **Coverage:** the forecasts start on 2024-01-21 (`fc_snow_sum` on 2024-12-09), so `fc_*` is empty before that. 2025 and 2026 are complete.
- **Shifts:** hours are converted to Seattle time and grouped with the cleaning rule (local time minus 8 hours), so daylight-saving days have 23 or 25 hours. Precipitation, snowfall and gusts cover the hour before their timestamp, so they are shifted to the hour where they start. A shift with any missing hour is left empty.
- **Features per shift:** `precip_sum` (mm), `precip_max_hr` (mm in one hour), `snow_sum` (cm), `temp_max` and `temp_min` (°C), `gust_max` (km/h).
