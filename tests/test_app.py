"""App behaviour tests: planning time, the historical average, model checks,
the hourly breakdown, and the Streamlit screen.

Run: uv run python -m unittest discover -s tests -v
"""

import functools
import json
import logging
import os
import re
import tempfile
import tomllib
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
os.environ.setdefault("CALLCAST_OFFLINE", "1")  # also in tests/__init__.py

import streamlit as st
from streamlit.testing.v1 import AppTest

# AppTest runs Streamlit outside its normal server context; suppress only its
# expected missing-context messages while keeping other warnings visible.


class _IgnoreBareModeWarning(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext" not in record.getMessage()


logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(
    _IgnoreBareModeWarning()
)

import chart_theme
from chart_theme import CSS_COLORS, STATION_COLORS
from station_areas import HOVER_COLOR, MAP_LOCK_JS
from text_blocks import plain
from app_data import (
    ROOT,
    add_likely_range,
    citywide_hourly,
    citywide_range,
    first_model_date,
    forecast_origin,
    hourly_allocation,
    known_history,
    learned_predictions,
    load_ranges,
    load_tables,
    reference_predictions,
    seattle_today,
    target_dates,
    validate_predictions,
)


@functools.cache
def tables():
    # Load the real CSVs once and share them between test classes.
    return load_tables()


class AppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stations, cls.counts, cls.features = tables()
        cls.ids = cls.stations.station.tolist()


def write_profile(root, history_through="2024-12-31", stations=("SFD 10",)):
    # Even 1/24 share per hour for each station, saved where the app looks for it.
    (root / "data").mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "Station": [s for s in stations for _ in range(24)],
            "shift_hour": list(range(24)) * len(stations),
            "share": [1 / 24] * 24 * len(stations),
            "history_through": history_through,
        }
    ).to_csv(root / "data/hourly_profile.csv", index=False)


# Planning time: the app may only use data a supervisor has on Thursday 08:00.


class PlanningTimeTests(AppTestCase):
    def test_planning_time_is_thursday_8am(self):
        # Saturday Oct 24 is planned Thursday Oct 22 at 08:00 Seattle time (PDT).
        self.assertEqual(
            str(forecast_origin("2026-10-24")), "2026-10-22 08:00:00-07:00"
        )
        # Planning times just after the daylight saving switches are in PST.
        for target in ["2026-03-09", "2026-11-03"]:
            self.assertTrue(str(forecast_origin(target)).endswith("-08:00"))

    def test_history_stops_at_last_finished_shift(self):
        # Wednesday Oct 21 is the last finished shift, and outage shifts are dropped.
        history = known_history(self.counts, "2026-10-24")
        self.assertEqual(history.ShiftDate.max(), pd.Timestamp("2026-10-21"))
        self.assertFalse(history.Masked.any())

    def test_future_counts_cannot_change_estimate(self):
        # Changing Thursday's and later counts to 999999 must not move the estimate.
        expected = reference_predictions(self.counts, "2025-06-07", self.ids)
        altered = self.counts.copy()
        altered.loc[altered.ShiftDate >= "2025-06-05", "calls"] = 999999
        pd.testing.assert_frame_equal(
            expected, reference_predictions(altered, "2025-06-07", self.ids)
        )

    def test_outage_shifts_are_ignored(self):
        # Outage counts aren't trusted, so changing them must not move the estimate.
        altered = self.counts.copy()
        altered.loc[altered.Masked, "calls"] = 999999
        pd.testing.assert_frame_equal(
            reference_predictions(self.counts, "2026-10-24", self.ids),
            reference_predictions(altered, "2026-10-24", self.ids),
        )

    def test_date_picker_range_follows_the_data(self):
        # Real data: first pick is a month after Jan 1 2022, fixed today is Oct 24.
        with mock.patch.dict(os.environ, {"CALLCAST_TODAY": "2026-10-24"}):
            self.assertEqual(
                target_dates(self.counts),
                (date(2022, 2, 1), date(2026, 10, 28), date(2026, 10, 24)),
            )
            # A smaller snapshot moves the default date inside its own range.
            year = pd.DataFrame({"ShiftDate": pd.date_range("2024-01-01", "2024-12-31")})
            self.assertEqual(target_dates(year)[2], date(2024, 12, 31))
            # Ten days is not enough history to pick any date.
            with self.assertRaises(ValueError):
                target_dates(year.iloc[:10])

    def test_picker_starts_when_the_model_can_serve(self):
        # The model is trained through 2025-12-31, so it can serve from January 3, 2026.
        self.assertEqual(first_model_date(), date(2026, 1, 3))
        with mock.patch.dict(os.environ, {"CALLCAST_TODAY": "2026-10-24"}):
            self.assertEqual(
                target_dates(self.counts, first_model_date()),
                (date(2026, 1, 3), date(2026, 10, 28), date(2026, 10, 24)),
            )
        # Without readable model metadata there is no limit.
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(first_model_date(Path(folder)))

    def test_picker_ends_at_the_last_forecast_date(self):
        # Forecast shifts after October 28 can be picked, 90 days ahead (to Jan 26, 2027);
        # October 28 stays the last day of counts.
        last_forecast = self.features.ShiftDate.max().date()
        self.assertEqual(last_forecast, date(2027, 1, 26))
        with mock.patch.dict(os.environ, {"CALLCAST_TODAY": "2026-10-24"}):
            self.assertEqual(
                target_dates(self.counts, first_model_date(), last_forecast),
                (date(2026, 1, 3), date(2027, 1, 26), date(2026, 10, 24)),
            )
        self.assertEqual(self.counts.ShiftDate.max().date(), date(2026, 10, 28))

    def test_default_date_uses_seattle_today_and_clamps_to_range(self):
        last_forecast = self.features.ShiftDate.max().date()
        # Unset env var asks Seattle local time (America/Los_Angeles).
        clean_env = {k: v for k, v in os.environ.items() if k != "CALLCAST_TODAY"}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            with mock.patch("app_data.pd.Timestamp.now", return_value=pd.Timestamp("2026-11-26 22:00-08:00")) as now:
                self.assertEqual(seattle_today(), date(2026, 11, 26))
                now.assert_called_once_with(tz="America/Los_Angeles")
        # Inside range, before range, and past last forecast day.
        for today_str, expected in [
            ("2026-11-26", date(2026, 11, 26)),
            ("2025-05-01", date(2026, 1, 3)),
            ("2028-02-01", date(2027, 1, 26)),
        ]:
            with mock.patch.dict(os.environ, {"CALLCAST_TODAY": today_str}):
                self.assertEqual(
                    target_dates(self.counts, first_model_date(), last_forecast)[2],
                    expected,
                )

    def test_forecast_planning_times(self):
        # October 29, 30 and 31 are planned on October 27, 28 and 29 at 08:00, and the
        # latest finished shifts are October 26, 27 and 28.
        for target, planned, latest in [
            ("2026-10-29", "2026-10-27", "2026-10-26"),
            ("2026-10-30", "2026-10-28", "2026-10-27"),
            ("2026-10-31", "2026-10-29", "2026-10-28"),
        ]:
            self.assertEqual(str(forecast_origin(target))[:16], planned + " 08:00")
            history = known_history(self.counts, target)
            self.assertEqual(history.ShiftDate.max(), pd.Timestamp(latest))


# Usual weekday average: the "Usual for this weekday" numbers shown next to each estimate.


class HistoricalReferenceTests(AppTestCase):
    def test_matches_hand_calculation(self):
        # Oct 24 (Saturday): average of the 247 earlier Saturdays per station.
        result = reference_predictions(self.counts, "2026-10-24", self.ids)
        values = result.set_index("Station").predicted_calls
        self.assertAlmostEqual(values["SFD 10"], 9620 / 247)  # 38.9
        self.assertAlmostEqual(values["SFD 2"], 8032 / 247)  # 32.5
        self.assertAlmostEqual(values.sum(), 306.4332, places=4)  # citywide
        self.assertEqual(len(result), 34)


# Prediction checks: bad model output is stopped before it reaches the screen.


class PredictionCheckTests(AppTestCase):
    def test_rejects_bad_prediction_tables(self):
        good = pd.DataFrame({"Station": self.ids, "predicted_calls": 2.5})
        self.assertEqual(len(validate_predictions(good, self.ids)), 34)
        # A missing station or a duplicated station.
        for bad in [good.iloc[:-1], pd.concat([good, good.iloc[:1]])]:
            with self.assertRaises(ValueError):
                validate_predictions(bad, self.ids)
        # Negative, infinite, blank (NaN) and pandas NA values.
        for number in [-1, np.inf, np.nan]:
            bad = good.copy()
            bad.loc[0, "predicted_calls"] = number
            with self.assertRaises(ValueError):
                validate_predictions(bad, self.ids)
        bad = good.assign(predicted_calls=pd.Series([pd.NA] * 34, dtype="Float64"))
        with self.assertRaises(ValueError):
            validate_predictions(bad, self.ids)


# Model connection: how the app loads predict.py and model_metadata.json.
# These use a fake predict.py in a temp folder, so they test the wiring, not a real model.


class ModelConnectionTests(AppTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.metadata = {
            "model_name": "fake model",
            "trained_through": "2025-12-29",
            "feature_columns": ["Station", "weekday"],
            "accepts_missing": False,
        }

    def save(self, predict_code=None, **changes):
        # Write predict.py and model_metadata.json, with optional metadata changes.
        if predict_code:
            (self.root / "predict.py").write_text(predict_code)
        (self.root / "model_metadata.json").write_text(
            json.dumps({**self.metadata, **changes})
        )

    def predict(self, target="2026-01-01"):
        return learned_predictions(self.features, target, self.ids, self.root)

    def test_missing_model_says_so(self):
        with self.assertRaisesRegex(ValueError, "not connected yet"):
            self.predict()

    def test_fake_model_runs_without_seeing_calls(self):
        # The fake model fails if it gets the answer column; otherwise 2.5 per station.
        self.save(
            "def predict_shift(features):\n"
            "    assert 'calls' not in features.columns\n"
            "    return features[['Station']].assign(predicted_calls=2.5)\n"
        )
        result, name = self.predict()
        self.assertEqual(result.predicted_calls.sum(), 34 * 2.5)
        self.assertEqual(name, "fake model")

    def test_predict_py_is_loaded_once_and_reloaded_when_edited(self):
        # predict.py keeps the model in memory, so a fresh copy per click would reload it.
        log = self.root / "loads.txt"
        code = (
            f"with open({str(log)!r}, 'a') as f:\n"
            "    f.write('x')\n"
            "def predict_shift(features):\n"
            "    return features[['Station']].assign(predicted_calls=1.0)\n"
        )
        self.save(code)
        self.predict()
        self.predict()
        self.assertEqual(log.read_text(), "x")
        # Editing the file (a different size) loads a fresh copy.
        self.save(code + "# edited\n")
        self.predict()
        self.assertEqual(log.read_text(), "xx")

    def test_rejects_model_trained_too_late(self):
        # Target Jan 1 is planned Dec 30, so training can only go up to Dec 29.
        self.save(
            "def predict_shift(f):\n    return None\n", trained_through="2025-12-31"
        )
        with self.assertRaisesRegex(ValueError, "planning time"):
            self.predict()

    def test_rejects_answer_as_input(self):
        self.save("def predict_shift(f):\n    return None\n", feature_columns=["calls"])
        with self.assertRaisesRegex(ValueError, "target information"):
            self.predict()

    def test_rejects_bad_metadata(self):
        self.save("def predict_shift(f):\n    raise RuntimeError('model crashed')\n")
        for key, value, error in [
            ("trained_through", None, ValueError),
            ("trained_through", "NaT", ValueError),
            ("model_name", "", ValueError),
            ("accepts_missing", "false", TypeError),  # text, not true/false
            ("feature_columns", ["Station", "Station"], ValueError),
        ]:
            self.save(**{key: value})
            with self.assertRaises(error, msg=f"{key}={value}"):
                self.predict()
        # With good metadata, a crash inside predict.py comes back as a readable error.
        self.save()
        with self.assertRaisesRegex(ValueError, "model crashed"):
            self.predict()


# Forecast shifts: the 90 shifts after the last day of counts. The first three use real
# recent calls; later ones use typical values from data/lag_fill_means.csv.


class ForecastTests(AppTestCase):
    def test_forecast_rows_are_added_without_calls(self):
        last_count = self.counts.ShiftDate.max()
        forecast = self.features[self.features.ShiftDate > last_count]
        self.assertEqual(len(forecast), 34 * 90)
        self.assertEqual(
            sorted(forecast.ShiftDate.dt.date.unique()),
            list(pd.date_range("2026-10-29", "2027-01-26").date),
        )
        self.assertTrue(forecast.calls.isna().all())
        self.assertEqual(set(forecast.Station), set(self.ids))
        # Every shift up to the last count day still has its real calls.
        self.assertTrue(
            self.features[self.features.ShiftDate <= last_count].calls.notna().all()
        )

    def test_long_range_estimates_follow_holidays(self):
        # With no recent calls, holidays still move the estimate: Christmas is quieter and
        # New Year's Eve busier than a usual shift on the same weekday.
        def gap(day):
            result, _ = learned_predictions(self.features, day, self.ids)
            usual = reference_predictions(self.counts, day, self.ids)
            return result.predicted_calls.sum() - usual.predicted_calls.sum()

        self.assertLess(gap("2026-12-25"), -20)
        self.assertGreater(gap("2026-12-31"), 20)

    def test_special_days_still_show_in_far_forecasts(self):
        totals = {
            day: learned_predictions(self.features, day, self.ids)[
                0
            ].predicted_calls.sum()
            for day in ["2026-12-30", "2026-12-31"]
        }
        # New Year's Eve is a special day in the model: calls are higher than the day before.
        self.assertGreater(totals["2026-12-31"], totals["2026-12-30"] + 30)

    def test_dates_past_the_last_forecast_say_so(self):
        # The day after the 90th forecast shift is out of range, not a data gap.
        for day in ["2027-01-27", "2027-06-01"]:
            with self.assertRaisesRegex(
                ValueError, "estimates go up to Jan 26, 2027"
            ):
                learned_predictions(self.features, day, self.ids)
        # A real gap inside the range keeps its own message.
        with self.assertRaisesRegex(ValueError, "call data for this shift has a gap"):
            learned_predictions(self.features, "2026-01-09", self.ids)

    def test_model_gives_34_stations_and_the_expected_totals(self):
        for target, total in [
            ("2026-10-29", 317.5),
            ("2026-10-30", 329.5),
            ("2026-10-31", 311.6),
        ]:
            result, _ = learned_predictions(self.features, target, self.ids)
            self.assertEqual(len(result), 34)
            self.assertEqual(round(result.predicted_calls.sum(), 1), total)

    def test_usual_number_comes_from_the_counts(self):
        # October 31 is a Saturday: the average of the 248 earlier Saturdays per station.
        usual = reference_predictions(self.counts, "2026-10-31", self.ids)
        values = usual.set_index("Station").predicted_calls
        self.assertEqual(len(usual), 34)
        self.assertAlmostEqual(values["SFD 10"], 9656 / 248)  # 38.9
        self.assertAlmostEqual(values["SFD 2"], 8071 / 248)  # 32.5
        self.assertAlmostEqual(values.sum(), 306.3992, places=4)  # citywide

    def test_dates_without_model_inputs_get_a_plain_message(self):
        # 13 shifts have a gap in the call data and no row in modeling_dataset.csv.
        have = set(self.features.ShiftDate.dt.date)
        days = pd.date_range("2026-01-03", "2026-10-28").date
        gaps = [day for day in days if day not in have]
        self.assertEqual(len(gaps), 13)
        self.assertIn(date(2026, 1, 9), gaps)
        for day in gaps:
            with self.assertRaisesRegex(
                ValueError, "call data for this shift has a gap"
            ):
                learned_predictions(self.features, day.isoformat(), self.ids)


# Likely range: how far real calls usually land from an estimate (data/prediction_ranges.csv).


class LikelyRangeTests(AppTestCase):
    def test_ranges_file_loads_and_covers_every_size(self):
        bands, city = load_ranges()
        self.assertEqual(len(bands), 5)
        self.assertEqual(bands.predicted_from.iloc[0], 0)
        self.assertTrue((bands.low < 0).all() and (bands.high > 0).all())
        self.assertTrue(city.low < 0 < city.high)
        # Bigger forecasts miss by more, so the ranges get wider.
        self.assertTrue((bands.high - bands.low).is_monotonic_increasing)

    def test_rejects_broken_range_files(self):
        good = pd.read_csv(ROOT / "data/prediction_ranges.csv")
        stations = good.level == "station"
        broken = {
            "a missing column": good.drop(columns="high"),
            "a gap between bands": good.assign(
                predicted_from=good.predicted_from.where(good.index != 2, 11)
            ),
            "a low above zero": good.assign(low=good.low.where(good.index != 0, 1.0)),
            "two citywide rows": pd.concat([good, good[~stations]]),
            "an unknown level": good.assign(
                level=good.level.where(good.index != 0, "x")
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            good.to_csv(root / "data/prediction_ranges.csv", index=False)
            load_ranges(root)  # the unbroken copy loads fine
            for name, frame in broken.items():
                frame.to_csv(root / "data/prediction_ranges.csv", index=False)
                with self.assertRaises(ValueError, msg=name):
                    load_ranges(root)

    def test_low_and_high_surround_each_estimate(self):
        bands, _ = load_ranges()
        # 1.0 would go below zero, 5.0 is the top of the first band, 39.3 is in the 20 to 40 band.
        view = pd.DataFrame({"predicted_calls": [1.0, 5.0, 5.1, 12.0, 39.3, 80.0]})
        result = add_likely_range(view, bands)
        self.assertTrue((result.low >= 0).all())
        self.assertTrue((result.low <= result.predicted_calls).all())
        self.assertTrue((result.high >= result.predicted_calls).all())
        self.assertEqual(result.low[0], 0)  # 1.0 - 2.4 is clamped to 0
        self.assertEqual(
            result.high[1], round(5.0 + bands.high[0])
        )  # 5.0 uses the first band
        self.assertEqual(
            result.high[2], round(5.1 + bands.high[1])
        )  # 5.1 uses the second
        self.assertEqual((result.low[4], result.high[4]), (32, 48))
        self.assertEqual(
            result.high[5], round(80.0 + bands.high.iloc[-1])
        )  # past the last band

    def test_citywide_range(self):
        _, city = load_ranges()
        self.assertEqual(citywide_range(313.6, city), (284, 346))


# Hourly breakdown: splitting one shift total into 24 hours.


class HourlyBreakdownTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_hours_add_up_to_shift_total(self):
        write_profile(self.root)
        result = hourly_allocation(self.root, "2026-10-24", "SFD 10", 30)
        self.assertAlmostEqual(result.expected_calls.sum(), 30)
        self.assertEqual(result.hour.iloc[0], "08:00")  # shift starts at 08:00

    def test_citywide_hours_add_every_station(self):
        write_profile(self.root, stations=("SFD 10", "SFD 2"))
        # Two stations stand for the city: 30 + 10 calls, and the file is read only once.
        with mock.patch("app_data.pd.read_csv", wraps=pd.read_csv) as read:
            result = citywide_hourly(self.root, "2026-10-24", {"SFD 10": 30, "SFD 2": 10})
        self.assertAlmostEqual(result.expected_calls.sum(), 40)
        self.assertEqual(len(result), 24)
        self.assertEqual(read.call_count, 1)
        with self.assertRaises(ValueError):
            citywide_hourly(self.root, "2026-10-24", {})

    def test_citywide_share_is_the_citys(self):
        # SFD 10 has all 30 calls at 08:00, SFD 2 spreads 10 calls evenly: 08:00 holds
        # (30 + 10/24) / 40 = 76% of the city's calls, not SFD 10's 100%.
        write_profile(self.root, stations=("SFD 10", "SFD 2"))
        path = self.root / "data/hourly_profile.csv"
        profile = pd.read_csv(path)
        profile.loc[profile.Station == "SFD 10", "share"] = [1.0] + [0.0] * 23
        profile.to_csv(path, index=False)
        result = citywide_hourly(self.root, "2026-10-24", {"SFD 10": 30, "SFD 2": 10})
        self.assertAlmostEqual(result.share.iloc[0], (30 + 10 / 24) / 40)
        self.assertAlmostEqual(result.share.sum(), 1)

    def test_rejects_profile_built_from_future_data(self):
        write_profile(self.root, history_through="2026-10-24")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            hourly_allocation(self.root, "2026-10-24", "SFD 10", 30)

    def test_skips_daylight_saving_shifts(self):
        # These shifts are 23 and 25 hours long, so a 24 hour profile doesn't fit.
        write_profile(self.root)
        for target, length in [("2026-03-07", 23), ("2025-11-01", 25)]:
            with self.assertRaisesRegex(ValueError, f"{length} hour"):
                hourly_allocation(self.root, target, "SFD 10", 30)

    def test_october_31_is_a_25_hour_shift(self):
        # The clocks go back on November 1, so the October 31 forecast shift is 25 hours long.
        write_profile(self.root)
        with self.assertRaisesRegex(ValueError, "25 hours long"):
            hourly_allocation(self.root, "2026-10-31", "SFD 10", 30)


# Screen: runs app.py like a browser would and reads what it shows.


def table_html(app):
    # The page has two components (the UI sounds and the station table); the table sends HTML.
    data = [json.loads(c.proto.json) for c in app.get("bidi_component")]
    return next(d["html"] for d in data if "html" in d)


def ranked(app):
    # The station table's rows as the browser gets them, read back from the table's HTML.
    html = table_html(app)
    rows = re.findall(
        r'<tr data-station="([^"]+)"( class=on)?>.*?</span></span>([\d.]+)</td>'
        r"<td>([^<]+)</td><td[^>]*>([^<]+)</td>(.*?)</tr>",
        html,
    )
    return pd.DataFrame(
        {
            "Station": [r[0] for r in rows],
            "Expected calls": [float(r[2]) for r in rows],
            "Likely range": [r[3] for r in rows],
            "Difference": [float(r[4].split()[-1]) for r in rows],
            "Last 4 weeks": [re.findall(r'class="bar[^"]*"[^>]*title="([^"]+)"', r[5]) for r in rows],
            "ticked": [bool(r[1]) for r in rows],
        }
    )


class ScreenTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"CALLCAST_TODAY": "2026-10-24"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_tab_names_and_explore_charts(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        # Explore has four sub-tabs (Citywide, Stations, Events, Weather); AppTest lists them in page order.
        self.assertEqual(
            [t.label for t in app.tabs],
            [
                "Prediction",
                "Explore",
                "Citywide",
                "Stations",
                "Events",
                "Weather",
                "About",
            ],
        )
        # Explore keeps the history charts; model performance stays out of the app.
        self.assertIn("without shifts that have missing data", shown(app.tabs[1]))
        text = " ".join(
            [x.value for tab in app.tabs for x in (*tab.markdown, *tab.subheader)]
            + [shown(tab) for tab in app.tabs]
        )
        self.assertNotIn("station_errors", text)
        # Our word for a gap in the feed stays off the screen.
        self.assertNotIn("outage", text)
        self.assertIn("missing data", text)

    def test_default_screen_and_station_compare(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        # No sidebar widgets: the date picker lives in the top bar.
        self.assertEqual(len(app.sidebar.date_input), 0)
        self.assertEqual(len(app.sidebar.selectbox), 0)
        self.assertEqual(len(app.sidebar.radio), 0)
        self.assertEqual(len(app.date_input), 1)
        # Default view: Oct 24, nothing picked, citywide and busiest station cards with the model's numbers.
        self.assertEqual(app.multiselect[0].value, [])
        self.assertEqual(len(app.metric), 2)
        self.assertEqual(app.metric[0].value, "312.8")
        # The card compares with the usual number: 312.8 - 306.4 = +6.3 (unrounded sums).
        self.assertEqual(app.metric[0].delta, "+6.3 vs usual")
        self.assertEqual(app.metric[1].label, "Biggest rise vs usual: SFD 39")
        self.assertEqual(app.metric[1].value, "11.5")
        table = ranked(app)
        # The table rounds each of the 34 stations to one decimal.
        self.assertAlmostEqual(table["Expected calls"].sum(), 312.8, delta=34 * 0.05)
        # The table covers all 34 stations, none below 0, and the city card is their sum.
        self.assertEqual(len(table), 34)
        self.assertTrue((table["Expected calls"] >= 0).all())
        self.assertAlmostEqual(
            float(app.metric[0].value.replace(",", "")),
            table["Expected calls"].sum(),
            delta=34 * 0.05,
        )
        # Busiest first, and every estimate sits inside its "low to high" range.
        self.assertTrue(table["Expected calls"].is_monotonic_decreasing)
        low_high = table["Likely range"].str.extract(r"^(\d+) to (\d+)$").astype(int)
        self.assertTrue((low_high[0] <= table["Expected calls"]).all())
        self.assertTrue((table["Expected calls"] <= low_high[1]).all())
        # Picking stations adds one card each, side by side; the top cards stay.
        app.multiselect[0].select("SFD 10").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "312.8")
        self.assertEqual(app.metric[2].label, "SFD 10 expected calls")
        self.assertEqual(app.metric[2].value, "40.0")
        self.assertEqual(app.metric[2].delta, "+1.1 vs usual")
        app.multiselect[0].select("SFD 2").run()
        self.assertEqual(
            [m.value for m in app.metric], ["312.8", "11.5", "40.0", "31.7"]
        )

    def test_theme_uses_light_top_bar_and_hides_sidebar(self):
        theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
        self.assertEqual(theme["theme"]["base"], "light")
        self.assertNotIn("sidebar", theme["theme"])
        css = (ROOT / "style.css").read_text()
        self.assertIn('[data-testid="stSidebar"]', css)
        self.assertIn("display: none !important", css)
        # The compare list's tags use the station colours of the rims and cards, in pick order.
        tags = re.findall(
            r'\[data-tag-index="(\d)"\] \{\s*background-color: var\(--cc-pick-(\d)\)', css
        )
        self.assertEqual(tags, [(str(i), str(i)) for i in range(len(STATION_COLORS))])
        self.assertEqual(
            [CSS_COLORS[f"pick-{i}"] for i in range(3)], STATION_COLORS
        )
        # style.css holds no colour of its own: every one is a --cc-* variable from chart_theme.
        # The motion settings and the tab icons are the only --cc-* names style.css sets itself.
        self.assertEqual(re.findall(r"#[0-9A-Fa-f]{3,8}\b|rgba?\(", css), [])
        names = set(re.findall(r"var\(--cc-([\w-]+)\)", css))
        own = set(re.findall(r"--cc-([\w-]+):", css))
        self.assertLessEqual(names - own, set(CSS_COLORS) | {"shadow"})
        # Top bar is a sky gradient with dark ink and never light text; the flat light blue
        # stays for the key's shaded-days swatch.
        self.assertEqual(chart_theme.HEADER_BG, chart_theme.LIGHT_BLUE)
        self.assertEqual(CSS_COLORS["header"], "#99CDD8")
        self.assertEqual(
            [CSS_COLORS[k] for k in ("sky-top", "sky-mid", "sky-low")],
            ["#E6F4FD", "#C6E3F8", "#A8D3F0"],
        )
        self.assertIn(
            "linear-gradient(115deg, var(--cc-sky-top) 0%, var(--cc-sky-mid) 40%, var(--cc-sky-low) 100%)",
            css,
        )
        # The cloud pictures come only from the git-ignored icon folder.
        pics = re.findall(r'url\("app/static/([\w/]+)/header_\w+\.png"\)', css)
        self.assertEqual(set(pics), {"weather_icons"})
        self.assertIn("static/weather_icons/", (ROOT / ".gitignore").read_text())
        # The clouds sit under the text, let clicks through, move only by transform and stay
        # still for reduced motion.
        self.assertIn("z-index: -1;\n    pointer-events: none;", css)
        frames = re.search(r"@keyframes cc-cloud-drift \{(.*?)\n\}", css, re.S).group(1)
        self.assertEqual(re.findall(r"([\w-]+):", frames), ["transform"])
        # The sun-with-cloud follows the title text on its line and never moves.
        sun = re.search(r"\.st-key-top_bar h3 \[data-heading-text\]::after \{(.*?)\}", css, re.S).group(1)
        self.assertIn("header_sun_cloud.png", sun)
        self.assertNotIn("animation", sun)
        self.assertNotIn("sun-drift", css)
        still = css[css.index("@media (prefers-reduced-motion: reduce) {\n  .st-key-top_bar::before"):]
        self.assertIn("animation: none;", still[: still.index("\n}")])
        # Pop-up windows get the very light blue panel, not the mint page colour.
        self.assertEqual(CSS_COLORS["panel"], "#F1FCFB")
        self.assertIn('[data-testid="stDialog"] [role="dialog"] {\n  background-color: var(--cc-panel)', css)
        top_bar_css = css[css.index(".st-key-top_bar") : css.index("/* Subtle button hover")]
        self.assertIn("var(--cc-ink)", top_bar_css)
        self.assertIn("var(--cc-header-muted)", top_bar_css)
        self.assertNotIn("--cc-light-text", top_bar_css)

    def test_map_hand_pointer_uses_keys_of_the_clickable_maps(self):
        # The rule names the Prediction map's box and both Explore maps, and it sets the cursor
        # on deck.gl's event layer, the element under the mouse.
        css = (ROOT / "style.css").read_text()
        rule = re.search(
            r":is\(([^)]*)\):has\(\.deck-tooltip([^{]*)\{\s*cursor: pointer", css
        )
        self.assertIsNotNone(rule)
        keys = re.findall(r"\.st-key-(\w+)", rule.group(1))
        self.assertEqual(keys, ["prediction_map_box", "city_value_map", "station_area_map"])
        self.assertIn(".deck-events-root", rule.group(2))
        source = (ROOT / "app.py").read_text()
        self.assertIn('st.container(key="prediction_map_box")', source)
        self.assertIn('EXPLORE_MAP_KEY = "station_area_map"', source)
        self.assertIn('CITY_MAP_KEY = "city_value_map"', source)

    def test_map_lock_script_targets_only_the_prediction_map(self):
        # The page loads the lock script, and it stops drags and the wheel only inside the
        # Prediction map's box, so the Explore map still moves.
        source = (ROOT / "app.py").read_text()
        self.assertIn("st.html(MAP_LOCK_JS, unsafe_allow_javascript=True)", source)
        self.assertIn(".st-key-prediction_map_box", MAP_LOCK_JS)
        self.assertNotIn("station_area_map", MAP_LOCK_JS)

    def test_map_height_comes_from_the_city_fit(self):
        # The map's set height is the one that shows the whole city at the fixed zoom; on wide
        # screens style.css lets it grow (flex) so the columns end level, and no rule pins it.
        source = (ROOT / "app.py").read_text()
        self.assertIn("height=fixed_view(boundary)[1],", source)
        css = (ROOT / "style.css").read_text()
        self.assertNotRegex(css, r"prediction_map_box[^{]*\{[^}]*height:\s*\d+px")
        grow = re.search(r"\.st-key-prediction_map_box > \[data-testid=\"stElementContainer\"\][^{]*\{([^}]*)\}", css)
        self.assertIn("flex: 1 0 auto", grow.group(1))
        # Below 1160 px the map's column would be narrower than the 360 px phone frame the zoom
        # fits, so the top stacks there, and the stretch only runs side by side.
        stack = css[css.index("@media (max-width: 1159px)"):]
        self.assertIn("display: contents", stack[:stack.index("\n}\n")])
        self.assertIn("@media (min-width: 1160px) {\n  [data-testid=\"stLayoutWrapper\"]:has(> .st-key-prediction_map_panel)", css)

    def test_streamlit_theme_matches_chart_theme(self):
        # .streamlit/config.toml cannot import Python, so check each colour against chart_theme.
        theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())["theme"]
        expected = {
            "primaryColor": chart_theme.ACCENT,
            "backgroundColor": chart_theme.PAGE_BG,
            "secondaryBackgroundColor": chart_theme.MINT,
            "textColor": chart_theme.INK,
            "linkColor": chart_theme.LINK,
            "borderColor": chart_theme.SAGE,
            "dataframeBorderColor": chart_theme.SAGE,
            "dataframeHeaderBackgroundColor": chart_theme.MINT,
            "yellowColor": chart_theme.OCHRE,
            "yellowBackgroundColor": chart_theme.OCHRE_TINT,
            "yellowTextColor": chart_theme.OCHRE_TEXT,
            "blueColor": chart_theme.DEEP_BLUE,
            "blueBackgroundColor": chart_theme.QUIET_TINT_SOFT,
            "blueTextColor": chart_theme.QUIET_TEXT,
            "redColor": chart_theme.TERRACOTTA_DARK,
            "redBackgroundColor": chart_theme.BUSY_TINT_SOFT,
            "redTextColor": chart_theme.TERRACOTTA_DARK,
            "greenColor": chart_theme.MOSS,
            "greenBackgroundColor": chart_theme.MOSS_TINT,
            "greenTextColor": chart_theme.MOSS_EDGE,
        }
        self.assertEqual({k: theme[k] for k in expected}, expected)
        self.assertEqual(
            theme["chartCategoricalColors"],
            [*STATION_COLORS, chart_theme.SAGE_GREY],
        )

    def test_screen_opens_on_fixed_today_date(self):
        with mock.patch.dict(os.environ, {"CALLCAST_TODAY": "2026-11-26"}):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.date_input[0].value, date(2026, 11, 26))
        self.assertIn("**Planned Tue Nov 24, 08:00**", [m.value for m in app.markdown])

    def test_screen_shows_no_technical_details(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        text = " ".join(
            [x.value for x in [*app.caption, *app.markdown, *app.info, *app.warning]]
            + [h.value for h in app.get("html") if "cc-text" in h.value or "cc-note" in h.value]
        )
        for word in ["Poisson", "GBM", "first pass", "Trained model", "reference mode"]:
            self.assertNotIn(word, text)
        self.assertEqual(len(app.code), 0)
        # The two planner links are there, and the repository link is not.
        self.assertIn("sfdlive.com", text)
        self.assertIn("seattlebreaking.com", text)
        self.assertNotIn("github.com", text)

    def test_a_date_with_a_data_gap_shows_a_plain_message(self):
        # Jan 9 has a gap in the call data: no numbers, no file names on the screen.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.date_input[0].set_value(date(2026, 1, 9)).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.warning, [])
        self.assertEqual(app.info, [])
        self.assertEqual(len(app.metric), 0)
        shown = " ".join(x.value for x in [*app.info, *app.caption])
        self.assertNotIn(".csv", shown)
        briefing = " ".join(h.value for h in app.get("html"))
        self.assertIn(
            "Friday has no call estimate because the call data for this shift has a gap.",
            briefing,
        )
        # The gap is said once, in the briefing; the note below only points to the Explore tab.
        self.assertEqual(briefing.count("no call estimate"), 1)
        self.assertNotIn("No estimate for this shift yet", briefing)
        self.assertIn("Past calls for each station are on the Explore tab.", briefing)
        # The same area map as other dates, in one pale shade, not Streamlit's dot map.
        self.assertEqual(len(app.get("map")), 0)
        spec = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
        areas = next(l for l in spec["layers"] if l.get("pickable"))
        self.assertEqual(len({tuple(f["properties"]["fill"]) for f in areas["data"]["features"]}), 1)

    def test_explore_charts_use_plain_names_in_order(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        # The hourly chart opens in a window now, so no Streamlit bar chart is on the page.
        self.assertEqual(len(app.get("vega_lite_chart")), 0)
        # Stations tab: calls by station, in number order (SFD 2 before SFD 18).
        stations = next(t for t in app.tabs if t.label == "Stations")
        by_station = json.loads(stations.get("plotly_chart")[0].proto.spec)
        shown = list(by_station["data"][0]["x"])
        self.assertEqual(len(shown), 34)
        self.assertEqual(shown, sorted(shown, key=lambda s: int(s.split()[-1])))
        # Citywide plotly charts: weekdays Monday to Sunday, categories in plain words.
        city = next(t for t in app.tabs if t.label == "Citywide")
        figures = [json.loads(c.proto.spec) for c in city.get("plotly_chart")]
        bars = [f for f in figures if f["data"][0]["type"] == "bar"]
        weekday = next(f for f in bars if f["data"][0].get("orientation") != "h")
        self.assertEqual(
            list(weekday["data"][0]["x"]),
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        )
        category = next(f for f in bars if f["data"][0].get("orientation") == "h")
        self.assertEqual(
            [t["name"] for t in category["data"]],
            ["Fire", "Medical", "Non-emergency", "Other"],
        )

    def test_likely_range_is_shown_for_city_station_and_table(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        bar_text = lambda: re.sub(
            r"<[^>]+>", " ", " ".join(h.value for h in app.tabs[0].get("html"))
        )
        # The city gets a range bar, and so does each picked station (SFD 10, expected 40.0).
        self.assertRegex(bar_text(), r"Low\s+283.*High\s+345")
        app.multiselect[0].select("SFD 10").run(timeout=60)
        self.assertRegex(bar_text(), r"Low\s+31.*High\s+50")
        intro = shown(app.tabs[0])
        self.assertIn(
            "Likely range about 4 in 5 similar station shifts and about 7 in 10 citywide totals "
            "ended up inside it. It shows how much the estimate can vary and is not a staffing "
            "recommendation.",
            intro,
        )
        self.assertIn(
            "Tick a row to compare that station, the same as clicking its area on the map. "
            "Click a station's Last 4 weeks bars for its history, or its Expected bar for "
            "the hourly breakdown. The download keeps the likely range as Low and High.",
            intro,
        )
        self.assertIn(
            "31 to 50", ranked(app).set_index("Station").loc["SFD 10", "Likely range"]
        )
        # No model scores or fold talk next to it.
        text = " ".join([intro] + [x.value for x in app.markdown])
        for word in ["MAE", "RMSE", "fold", "cross-valid", "cross valid"]:
            self.assertNotIn(word, text)
        # The range follows the date: Oct 31 has its own estimate and range.
        app.date_input[0].set_value(date(2026, 10, 31)).run(timeout=60)
        self.assertRegex(bar_text(), r"Low\s+282.*High\s+344")

    def test_model_error_is_looked_up_and_shown_once(self):
        # A model failure should give one warning, and the page should ask the model once.
        # Estimates are kept per date, so start without earlier tests' copies.
        st.cache_data.clear()
        with mock.patch(
            "app_data.learned_predictions", side_effect=ValueError("model broke")
        ) as lookup:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual([w.value for w in app.warning], ["model broke"])

    def test_date_picker_range_on_screen(self):
        # The picker starts where the model can serve and ends at the last forecast date.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        picker = app.date_input[0]
        self.assertEqual(
            (picker.min, picker.max), (date(2026, 1, 3), date(2027, 1, 26))
        )
        # The top bar shows the planning time and how far ahead estimates go.
        self.assertIn("**Planned Thu Oct 22, 08:00**", [m.value for m in app.markdown])
        self.assertIn(
            "Estimates are available up to Jan 26, 2027.",
            [c.value for c in app.caption],
        )

    def test_forecast_dates_are_labelled_and_checked_on_screen(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for day, total in [
            (date(2026, 10, 29), "317.5"),
            (date(2026, 10, 30), "329.5"),
            (date(2026, 10, 31), "311.6"),
        ]:
            app.date_input[0].set_value(day).run(timeout=60)
            self.assertFalse(app.exception, str(day))
            self.assertEqual(app.metric[0].value, total)
            self.assertEqual(len(ranked(app)), 34)
            self.assertIn("Forecast", app.subheader[0].value)
            self.assertIn("Forecast: this shift is after the last day of data", shown(app))
            # The first three forecast shifts still have real recent calls.
            self.assertNotIn("long-range", app.subheader[0].value)
        # October 31 is 25 hours long, so the hourly window says so instead of drawing.
        click_table(app, "hourly", "SFD 10")
        notes = " ".join(x.value for d in app.get("dialog") for x in d.info)
        self.assertIn("25 hours long", notes)
        # The last day with real counts is not labelled as a forecast.
        app.date_input[0].set_value(date(2026, 10, 28)).run(timeout=60)
        self.assertNotIn("Forecast", app.subheader[0].value)
        self.assertNotIn("Forecast:", shown(app))

    def test_long_range_dates_are_labelled_on_screen(self):
        # From Nov 1 (4 shifts after the last counts) to Jan 26, 2027 the estimate uses
        # typical recent-call values, and the screen says so once.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for day in [date(2026, 11, 1), date(2026, 12, 25), date(2027, 1, 26)]:
            app.date_input[0].set_value(day).run(timeout=60)
            self.assertFalse(app.exception, str(day))
            self.assertIn("Forecast", app.subheader[0].value)
            self.assertIn("long-range estimate", app.subheader[0].value)
            self.assertEqual(shown(app).count("Long-range estimate:"), 1, str(day))
            self.assertEqual(len(ranked(app)), 34)
        # Every estimate says that weather and events are not included.
        self.assertIn("does not include weather", shown(app))
        # A shift with real recent calls has no long-range note and the usual range wording.
        app.date_input[0].set_value(date(2026, 10, 31)).run(timeout=60)
        captions = shown(app)
        self.assertNotIn("Long-range estimate", captions)
        self.assertIn("about 4 in 5 similar station shifts", captions)
        self.assertIn("about 7 in 10 citywide totals", captions)

    def test_explanation_of_far_dates_is_on_the_about_tab(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        about = next(t for t in app.tabs if t.label == "About")
        text = shown(about)
        self.assertIn("Dates far ahead", text)
        self.assertIn("Estimates go up to 90 days after the last data", text)

    def test_explore_says_which_shifts_it_uses(self):
        # Explore follows the planning time; past the data, every date uses the same history.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)

        def explore_note_and_table():
            explore = next(t for t in app.tabs if t.label == "Explore")
            city = next(t for t in app.tabs if t.label == "Citywide")
            # A plain caption normally; an info box when the date is past the data.
            notes = [shown(explore)] + [i.value for i in explore.info]
            note = next(
                n for n in notes if "Citywide and Stations use finished shifts" in n
            )
            return note, city.dataframe[0].value

        def explore_info():
            explore = next(t for t in app.tabs if t.label == "Explore")
            return [
                i.value for i in explore.info if "after the last day of data" in i.value
            ]

        for day, expected, boxed in [
            # Jan 9 is an outage shift, so the last usable one before the planning time is Jan 8;
            # that is not the end of the data.
            (
                date(2026, 1, 12),
                "up to Thu Jan 08, 2026 (the last one finished by the planning time)",
                False,
            ),
            (
                date(2026, 10, 31),
                "up to Wed Oct 28, 2026 (the last one finished by the planning time)",
                False,
            ),
            (
                date(2026, 11, 1),
                "up to Wed Oct 28, 2026, so every later date shows the same history",
                True,
            ),
        ]:
            app.date_input[0].set_value(day).run(timeout=60)
            self.assertFalse(app.exception, str(day))
            self.assertIn(expected, explore_note_and_table()[0])
            # Past the data the note stands out as an info box, only inside Explore.
            self.assertEqual(bool(explore_info()), boxed, str(day))
        prediction = next(t for t in app.tabs if t.label == "Prediction")
        self.assertFalse(
            any("after the last day of data" in i.value for i in prediction.info)
        )
        self.assertIn(
            "The estimate for this date is on the Prediction tab.", explore_info()[0]
        )
        _, nov = explore_note_and_table()
        app.date_input[0].set_value(date(2027, 1, 20)).run(timeout=60)
        pd.testing.assert_frame_equal(nov, explore_note_and_table()[1])

    def test_explore_charts_explain_how_to_read_them(self):
        # Every Explore chart has a one-line note on what it shows and how to read it.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        city = shown(next(t for t in app.tabs if t.label == "Citywide"))
        for phrase in [
            "the top edge is all calls",
            "busier days of the week",
            "seasons and longer-term changes",
            "split by type",
        ]:
            self.assertIn(phrase, city)
        stations = shown(next(t for t in app.tabs if t.label == "Stations"))
        self.assertIn("Bars above the dashed line are busier", stations)


def click_area(app, name):
    # AppTest cannot click a map, so write the click under the current rotated layer id while
    # keeping earlier layer keys, matching Streamlit's DeckGlJsonChart payload.
    spec = json.loads(app.tabs[0].get("deck_gl_json_chart")[0].proto.json)
    layer_id = next(l["id"] for l in spec["layers"] if l.get("pickable"))
    prev = (
        (app.session_state["prediction_map"] or {})
        .get("selection", {})
        .get("objects", {})
    )
    objects = {**prev, layer_id: [{"properties": {"Station": name}}]}
    app.session_state["prediction_map"] = {
        "selection": {"objects": objects, "indices": {}}
    }
    return app.run(timeout=60)


def ticked(app):
    # The stations on the rows the table shows ticked, busiest first.
    table = ranked(app)
    return list(table.loc[table["ticked"], "Station"])


def click_table(app, act, name):
    # AppTest cannot click the table, so send the click the browser sends: the action and the
    # clicked row's station. Like the browser's, it lasts for this one run.
    app.session_state["ranked_table"] = {"action": {"act": act, "station": name}}
    return app.run(timeout=60)


def tick_row(app, name):
    return click_table(app, "tick", name)


def windows(app):
    return [d.proto.dialog.title for d in app.get("dialog")]


def shown(node):
    # Every caption and text block under node, as one line of plain words.
    return " ".join([c.value for c in node.caption] + [plain(h.value) for h in node.get("html")])


class PredictionTabTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"CALLCAST_TODAY": "2026-10-24"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_view_shows_city_map_watch_list_and_call_types(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        headings = [m.value for m in app.markdown]
        self.assertIn("##### Stations to watch", headings)
        self.assertIn("##### Call type mix (average calls per shift)", headings)
        captions = [c.value for c in app.caption]
        # Oct 24: no station is far enough above usual, and the screen says so.
        self.assertIn("No station is clearly busier than usual for this shift.", shown(app))
        self.assertIn("From past shifts in this season, not a forecast.", shown(app))
        self.assertIn(
            "Point for details; click up to 3 to compare.",
            captions,
        )
        # Nothing picked: the call type mix is the city's. The long history chart is gone from
        # the bottom: each station's history opens in a window from the table.
        mix = next(
            d.value for d in app.dataframe if list(d.value.columns) == ["Citywide"]
        )
        self.assertEqual(list(mix.index), ["Medical", "Fire", "Non-emergency", "Other"])
        self.assertFalse(
            any("history known at planning time" in s.value for s in app.subheader)
        )
        self.assertEqual(len(ranked(app)), 34)
        # Top briefing and range bars render inside HTML elements without percentages.
        html_list = [h.value for h in app.tabs[0].get("html")]
        html_blob = " ".join(html_list)
        briefing = next(h for h in html_list if "Saturday looks" in h)
        self.assertIn(
            "Saturday looks a little busier than usual, spread across the city.",
            briefing,
        )
        self.assertNotIn("%", briefing)
        self.assertIn("Low", html_blob)
        self.assertIn("High", html_blob)
        self.assertIn("Expected", html_blob)
        self.assertIn("Usual", html_blob)

    def test_picks_survive_a_date_with_a_data_gap(self):
        # Jan 9, 2026 has no estimate and no compare cards; the picks must still be there after.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.multiselect[0].set_value(["SFD 10", "SFD 2"]).run(timeout=60)
        app.date_input[0].set_value(date(2026, 1, 9)).run(timeout=60)
        app.date_input[0].set_value(date(2026, 10, 24)).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.multiselect[0].value, ["SFD 10", "SFD 2"])
        self.assertIn("Comparing SFD 10, SFD 2", [s.value for s in app.subheader])

    def test_freshness_line_gives_the_end_of_the_last_finished_shift(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        # Planning time is Thursday Oct 22 08:00, when the Wednesday shift has just ended.
        self.assertIn(
            "Data through Thu Oct 22 08:00, the last shift finished by the planning time.",
            shown(app),
        )

    def test_map_is_shaded_by_difference_and_has_the_hover_numbers(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        spec = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
        areas = next(l for l in spec["layers"] if l.get("pickable"))
        self.assertEqual(len(areas["data"]["features"]), 34)
        props = areas["data"]["features"][0]["properties"]
        # Plain values, so the tooltip can lay them out (a <br/> inside a value would print as text).
        self.assertRegex(props["expected"], r"^[\d.]+$")
        self.assertRegex(props["range"], r"^\d+ to \d+$")
        self.assertRegex(props["usual"], r"^[\d.]+$")
        # Flat, north-up Prediction map that cannot move, on a pale Carto basemap with nearby
        # town names, white borders, and station labels inside.
        self.assertFalse(areas["extruded"])
        self.assertEqual(areas["getLineColor"], [255, 255, 255, 255])
        view = spec["initialViewState"]
        self.assertEqual([view["pitch"], view["bearing"]], [0, 0])
        self.assertEqual([view["minZoom"], view["maxZoom"]], [view["zoom"]] * 2)
        self.assertIs(view["dragPan"], False)
        self.assertIs(view["scrollZoom"], False)
        self.assertIn("positron-nolabels", spec["mapStyle"])
        self.assertIn("nearby-places", [l["id"] for l in spec["layers"]])
        self.assertEqual(props["fill"][3], 255)
        self.assertIn(
            props["status"],
            [
                "Quieter than usual",
                "A little quieter than usual",
                "About usual",
                "A little busier than usual",
                "Busier than usual",
            ],
        )
        labels = next(l for l in spec["layers"] if l["id"] == "prediction-area-labels")
        self.assertEqual(len(labels["data"]), 34)
        texts = [row["text"] for row in labels["data"]]
        self.assertIn("10\n40.0", texts)
        self.assertIn("5", texts)
        fills = {
            f["properties"]["Station"]: f["properties"]["fill"]
            for f in areas["data"]["features"]
        }
        self.assertGreater(len({tuple(f) for f in fills.values()}), 10)
        table = ranked(app).set_index("Station")["Difference"]
        busiest, quietest = fills[table.idxmax()], fills[table.idxmin()]
        self.assertGreater(busiest[0] - busiest[2], 0)
        self.assertLess(quietest[0] - quietest[2], 0)

    def test_two_stations_are_compared_side_by_side(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        app.multiselect[0].select("SFD 10").select("SFD 2").run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(
            [m.label for m in app.metric][2:],
            ["SFD 10 expected calls", "SFD 2 expected calls"],
        )
        bar_text = re.sub(
            r"<[^>]+>", " ", " ".join(h.value for h in app.tabs[0].get("html"))
        )
        self.assertRegex(bar_text, r"Low\s+31\s+Usual\s+38\.9\s+High\s+50")
        self.assertRegex(bar_text, r"Low\s+24\s+Usual\s+32\.5\s+High\s+40")
        self.assertIn("Comparing SFD 10, SFD 2", [s.value for s in app.subheader])
        mix = next(
            d.value
            for d in app.dataframe
            if list(d.value.columns) == ["SFD 10", "SFD 2"]
        )
        self.assertEqual(len(mix), 4)
        # Both compare cards include range bars and coloured top borders, and the map draws picked rims.
        html_joined = " ".join(h.value for h in app.get("html"))
        for i in range(2):
            self.assertIn(
                f".st-key-card_compare_{i} {{ border-top: 4px solid {STATION_COLORS[i]}",
                html_joined,
            )
        spec = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
        rim = next(l for l in spec["layers"] if l["id"] == "picked-rim")
        self.assertEqual(len(rim["data"]), 2)

    def test_map_clicks_pick_unpick_and_stop_at_three(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for name in ["SFD 5", "SFD 10", "SFD 2"]:
            click_area(app, name)
        self.assertEqual(app.multiselect[0].value, ["SFD 5", "SFD 10", "SFD 2"])
        self.assertEqual(len(app.metric), 5)
        # A fourth area is refused with a plain message, and the three stay picked.
        click_area(app, "SFD 3")
        self.assertEqual(app.multiselect[0].value, ["SFD 5", "SFD 10", "SFD 2"])
        self.assertTrue(
            any(
                "Up to 3 stations can be compared at once" in w.value
                for w in app.warning
            )
        )
        # Clicking a picked area again removes it, which makes room.
        click_area(app, "SFD 10")
        self.assertEqual(app.multiselect[0].value, ["SFD 5", "SFD 2"])
        click_area(app, "SFD 3")
        self.assertEqual(app.multiselect[0].value, ["SFD 5", "SFD 2", "SFD 3"])
        self.assertFalse(app.exception, [e.message for e in app.exception])

    def test_event_warning_shows_for_the_picked_station_only(self):
        # Oct 25 is a Seahawks home game: the warning belongs to SFD 10, not to SFD 2.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        app.date_input[0].set_value(date(2026, 10, 25)).run(timeout=60)
        self.assertEqual(len(app.warning), 0)
        app.multiselect[0].select("SFD 2").run(timeout=60)
        self.assertEqual(len(app.warning), 0)
        app.multiselect[0].select("SFD 10").run(timeout=60)
        self.assertEqual(len(app.warning), 1)
        self.assertIn("Seahawks home game", app.warning[0].value)
        self.assertIn(
            "Event warnings are informational and are not included in the forecast.",
            shown(app),
        )
        # Said once: the table note then names only weather, not events again.
        text = shown(app)
        self.assertEqual(text.count("not included in the forecast"), 1)
        self.assertIn("The estimate does not include weather.", text)
        self.assertNotIn("events such as games", text)

    def test_watch_list_names_stations_clearly_above_usual(self):
        # Jan 17, 2027 has 8 stations above the rule; the list shows the 5 biggest gaps.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        app.date_input[0].set_value(date(2027, 1, 17)).run(timeout=60)
        items = [m.value for m in app.markdown if m.value.startswith("- **")]
        self.assertEqual(len(items), 5)
        self.assertTrue(all("vs " in i and i.endswith("usual") for i in items))
        self.assertFalse(any("%" in i for i in items))

    def test_hourly_breakdown_opens_in_a_window_for_the_clicked_station(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(any(e.label == "Hourly breakdown" for e in app.expander))
        app.multiselect[0].select("SFD 2").run(timeout=60)
        # The window follows the clicked row, not the first pick.
        click_table(app, "hourly", "SFD 25")
        self.assertEqual(windows(app), ["SFD 25: hourly breakdown, Sat Oct 24"])
        self.assertEqual(len(app.get("dialog")[0].get("vega_lite_chart")), 1)
        app.run(timeout=60)
        self.assertEqual(windows(app), [])
        app.button(key="city_hourly").click().run(timeout=60)
        self.assertEqual(windows(app), ["Citywide: hourly breakdown, Sat Oct 24"])
        self.assertFalse(app.exception, [e.message for e in app.exception])

    def test_new_screen_keeps_plain_words_and_the_single_long_range_note(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        app.date_input[0].set_value(date(2026, 11, 2)).run(timeout=60)
        app.multiselect[0].select("SFD 10").run(timeout=60)
        tab = app.tabs[0]
        text = " ".join(
            [x.value for x in [*tab.markdown, *tab.subheader, *tab.warning]] + [shown(tab)]
        )
        for word in ["MAE", "fold", "model", "Poisson"]:
            self.assertNotIn(word, text)
        self.assertEqual(shown(app).count("Long-range estimate:"), 1)

    def test_table_rows_pick_like_the_map_and_stay_in_sync(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        tick_row(app, "SFD 10")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        # A ticked row picks the station: list, compare card, map rim and details follow.
        self.assertEqual(app.multiselect[0].value, ["SFD 10"])
        self.assertEqual(app.metric[2].label, "SFD 10 expected calls")
        spec = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
        self.assertEqual(
            len(next(l for l in spec["layers"] if l["id"] == "picked-rim")["data"]), 1
        )
        self.assertIn(["SFD 10"], [list(d.value.columns) for d in app.dataframe])
        # A map click and a list change tick and untick the matching rows.
        click_area(app, "SFD 2")
        self.assertEqual(ticked(app), ["SFD 10", "SFD 2"])
        app.multiselect[0].set_value(["SFD 2"]).run(timeout=60)
        self.assertEqual(ticked(app), ["SFD 2"])
        # Unticking the row unpicks the station.
        tick_row(app, "SFD 2")
        self.assertEqual(app.multiselect[0].value, [])
        self.assertEqual(ticked(app), [])

    def test_a_fourth_table_row_is_refused_and_unticked(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for name in ["SFD 10", "SFD 2", "SFD 25"]:
            tick_row(app, name)
        tick_row(app, "SFD 5")
        self.assertEqual(app.multiselect[0].value, ["SFD 10", "SFD 2", "SFD 25"])
        self.assertTrue(any("SFD 5 was not added" in w.value for w in app.warning))
        # The table is drawn again with only the 3 picks ticked.
        self.assertEqual(sorted(ticked(app)), ["SFD 10", "SFD 2", "SFD 25"])
        # Unticking a pick then makes room, and SFD 5 is not slipped in.
        tick_row(app, "SFD 10")
        self.assertEqual(app.multiselect[0].value, ["SFD 2", "SFD 25"])
        tick_row(app, "SFD 5")
        self.assertEqual(app.multiselect[0].value, ["SFD 2", "SFD 25", "SFD 5"])
        self.assertFalse(app.exception, [e.message for e in app.exception])

    def test_table_ticks_follow_the_picks_across_dates(self):
        # The rows change order with the date, and a gap date has no table at all.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        tick_row(app, "SFD 5")
        tick_row(app, "SFD 39")
        for day in [date(2026, 10, 31), date(2026, 1, 9), date(2026, 10, 25)]:
            app.date_input[0].set_value(day).run(timeout=60)
            self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.multiselect[0].value, ["SFD 5", "SFD 39"])
        self.assertEqual(sorted(ticked(app)), ["SFD 39", "SFD 5"])
        tick_row(app, "SFD 5")
        self.assertEqual(app.multiselect[0].value, ["SFD 39"])

    def test_table_shows_bars_arrows_and_the_last_4_weeks(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        table = ranked(app).set_index("Station")
        # SFD 10 on every day from Sep 24 to Oct 21, all known by Thursday Oct 22.
        bars = table.loc["SFD 10", "Last 4 weeks"]
        self.assertEqual(len(bars), 28)
        self.assertEqual([bars[0], bars[-1]], ["Thu Sep 24: 40 calls", "Wed Oct 21: 40 calls"])
        self.assertEqual(
            bars[2::7],
            ["Sat Sep 26, same weekday: 36 calls", "Sat Oct 03, same weekday: 46 calls",
             "Sat Oct 10, same weekday: 37 calls", "Sat Oct 17, same weekday: 41 calls"],
        )
        self.assertTrue(all(len(b) == 28 for b in table["Last 4 weeks"]))
        html = table_html(app)
        # Only the Saturdays say "same weekday", 4 in each of the 34 rows; no bar has a dark top.
        days = re.findall(r'title="(\w+) \w+ \d+, same weekday', html)
        self.assertEqual((len(days), set(days)), (4 * 34, {"Sat"}))
        self.assertNotIn("bar day", html)
        for words in ["Last 4 weeks", "Sep 24 to Oct 21", "vs usual", "Likely range"]:
            self.assertIn(words, html)
        # No numbers column, no week-by-week column and no usual column.
        for words in ["Last 4 Saturdays", "Week by week", "Usual for this weekday"]:
            self.assertNotIn(words, html)

    def test_last_4_weeks_follow_the_date_and_mark_estimates(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        # Friday Nov 6 shows Oct 7 to Nov 3: real calls up to Oct 28, estimates after it.
        app.date_input[0].set_value(date(2026, 11, 6)).run(timeout=60)
        bars = ranked(app).set_index("Station").loc["SFD 10", "Last 4 weeks"]
        self.assertEqual(bars[2::7][:3], [
            "Fri Oct 09, same weekday: 50 calls", "Fri Oct 16, same weekday: 48 calls",
            "Fri Oct 23, same weekday: 49 calls",
        ])
        self.assertTrue(bars[21].startswith("Wed Oct 28: ") and bars[21].endswith(" calls"))
        self.assertRegex(bars[22], r"^Thu Oct 29: about \d+\.\d, an estimate$")
        self.assertTrue(all(b.endswith("an estimate") for b in bars[22:]))
        html = table_html(app)
        self.assertIn("Oct 07 to Nov 03, estimated after Oct 28", html)
        # Three weeks later only the first day, Oct 28, is real; the newest is Nov 24.
        app.date_input[0].set_value(date(2026, 11, 27)).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        bars = ranked(app).set_index("Station").loc["SFD 10", "Last 4 weeks"]
        self.assertEqual([bars[0].split(":")[0], bars[-1].split(":")[0]], ["Wed Oct 28", "Tue Nov 24"])
        self.assertTrue(bars[0].endswith(" calls"))
        self.assertTrue(all(b.endswith("an estimate") for b in bars[1:]))

    def test_last_4_weeks_leave_a_data_gap_empty(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        # Thursday Oct 1 shows Sep 1 to Sep 28; Sep 17 is an outage shift with no data.
        app.date_input[0].set_value(date(2026, 10, 1)).run(timeout=60)
        bars = ranked(app).set_index("Station").loc["SFD 10", "Last 4 weeks"]
        self.assertEqual((len(bars), bars[16]), (28, "Thu Sep 17, same weekday: no data"))
        self.assertIn('class="bar gap" title="Thu Sep 17, same weekday: no data"', table_html(app))

    def test_a_history_click_opens_that_rows_window_once(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        tick_row(app, "SFD 2")
        click_table(app, "history", "SFD 10")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(windows(app), ["SFD 10: history known at planning time"])
        summary = app.get("dialog")[0].get("html")[0].value
        text = [plain(p) for p in re.findall(r'<p class="cc-lead">(.*?)</p>', summary)]
        self.assertEqual(text[0], "Last 4 Saturdays: 36, 46, 37, 41.")
        self.assertTrue(text[1].startswith("Expected 40.0 on Sat Oct 24 (likely "))
        # Opening the window does not pick SFD 10 or drop SFD 2, and the next run (a list
        # change, or the run after the window closed) opens nothing.
        self.assertEqual(app.multiselect[0].value, ["SFD 2"])
        self.assertEqual(ticked(app), ["SFD 2"])
        app.multiselect[0].set_value(["SFD 2", "SFD 5"]).run(timeout=60)
        self.assertEqual(windows(app), [])

    def test_rows_open_their_own_window_in_any_order(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        steps = [
            ("history", "SFD 25"), ("tick", "SFD 10"), ("history", "SFD 5"),
            ("hourly", "SFD 10"), ("tick", "SFD 2"), ("tick", "SFD 31"), ("tick", "SFD 17"),
            ("history", "SFD 17"), (None, None), ("history", "SFD 17"), ("hourly", "SFD 31"),
            ("tick", "SFD 10"), ("history", "SFD 2"),
        ]
        for act, name in steps:
            if act is None:
                app.run(timeout=60)
                self.assertEqual(windows(app), [])
                continue
            click_table(app, act, name)
            self.assertFalse(app.exception, [e.message for e in app.exception])
            if act == "tick":
                self.assertEqual(windows(app), [], name)
            else:
                word = "history known" if act == "history" else "hourly breakdown"
                self.assertEqual(len(windows(app)), 1)
                self.assertTrue(windows(app)[0].startswith(f"{name}: {word}"), windows(app))
        # SFD 17 was refused as a 4th pick; unticking SFD 10 left SFD 2 and SFD 31.
        self.assertEqual(app.multiselect[0].value, ["SFD 2", "SFD 31"])
        self.assertEqual(ticked(app), ["SFD 2", "SFD 31"])
        # After a date change the rows change order, and a click still opens its own row.
        app.date_input[0].set_value(date(2026, 11, 27)).run(timeout=60)
        click_table(app, "history", "SFD 31")
        self.assertEqual(windows(app), ["SFD 31: history known at planning time"])

    def _pickable_id(self, tab):
        spec = json.loads(tab.get("deck_gl_json_chart")[0].proto.json)
        return next(l["id"] for l in spec["layers"] if l.get("pickable"))

    def _tap_layer(self, app, map_key, layer_id, names):
        objects = {layer_id: [{"properties": {"Station": n}} for n in names]}
        app.session_state[map_key] = {"selection": {"objects": objects, "indices": {}}}
        return app.run(timeout=60)

    def test_rotated_map_layer_clears_refused_click_and_list_unpick(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for name in ["SFD 10", "SFD 2", "SFD 25"]:
            lid = self._pickable_id(app.tabs[0])
            self._tap_layer(app, "prediction_map", lid, [name])
        first_id = self._pickable_id(app.tabs[0])
        self._tap_layer(app, "prediction_map", first_id, ["SFD 5"])
        # Refusing the 4th click rotates the layer id so Streamlit clears the clicked highlight.
        self.assertNotEqual(self._pickable_id(app.tabs[0]), first_id)
        self.assertEqual(app.multiselect[0].value, ["SFD 10", "SFD 2", "SFD 25"])
        self.assertTrue(any("SFD 5 was not added" in w.value for w in app.warning))
        # Removing SFD 10 from the list and clicking it again on the map picks it cleanly.
        app.multiselect[0].set_value(["SFD 2", "SFD 25"]).run(timeout=60)
        self._tap_layer(
            app, "prediction_map", self._pickable_id(app.tabs[0]), ["SFD 10"]
        )
        self.assertEqual(app.multiselect[0].value, ["SFD 2", "SFD 25", "SFD 10"])

    def test_two_quick_clicks_in_one_layer_generation_pick_both(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        lid = self._pickable_id(app.tabs[0])
        self._tap_layer(app, "prediction_map", lid, ["SFD 10"])
        self.assertEqual(app.session_state["map_clicked"], ["SFD 10"])
        self.assertNotEqual(self._pickable_id(app.tabs[0]), lid)
        # A second click before the browser resets the layer still sends the old layer id (current absent).
        self._tap_layer(app, "prediction_map", lid, ["SFD 10", "SFD 2"])
        self.assertEqual(app.multiselect[0].value, ["SFD 10", "SFD 2"])
        # Non-click reruns (e.g. changing date) leave picks untouched.
        app.date_input[0].set_value(date(2026, 10, 31)).run(timeout=60)
        self.assertEqual(app.multiselect[0].value, ["SFD 10", "SFD 2"])

    def test_map_picks_and_click_state_survive_gap_date(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        lid = self._pickable_id(app.tabs[0])
        self._tap_layer(app, "prediction_map", lid, ["SFD 10"])
        app.date_input[0].set_value(date(2026, 1, 9)).run(timeout=60)
        app.date_input[0].set_value(date(2026, 10, 24)).run(timeout=60)
        self.assertEqual(app.multiselect[0].value, ["SFD 10"])
        # After returning from a gap date, clicking SFD 10 unpicks it.
        self._tap_layer(
            app, "prediction_map", self._pickable_id(app.tabs[0]), ["SFD 10"]
        )
        self.assertEqual(app.multiselect[0].value, [])

    def test_explore_stations_map_refuses_fourth_click_and_keeps_three(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        stations = next(t for t in app.tabs if t.label == "Stations")
        objects = {}
        for name in ["SFD 10", "SFD 2", "SFD 25"]:
            lid = self._pickable_id(stations)
            objects[lid] = [{"properties": {"Station": name}}]
            app.session_state["station_area_map"] = {
                "selection": {"objects": dict(objects), "indices": {}}
            }
            app.run(timeout=60)
            stations = next(t for t in app.tabs if t.label == "Stations")
        before_id = self._pickable_id(stations)
        objects[before_id] = [{"properties": {"Station": "SFD 5"}}]
        app.session_state["station_area_map"] = {
            "selection": {"objects": dict(objects), "indices": {}}
        }
        app.run(timeout=60)
        stations = next(t for t in app.tabs if t.label == "Stations")
        self.assertNotEqual(self._pickable_id(stations), before_id)
        self.assertTrue(any("SFD 5 was not added" in w.value for w in stations.warning))
        self.assertIn(
            "Comparing SFD 10, SFD 2, SFD 25", [s.value for s in stations.subheader]
        )
        objects[self._pickable_id(stations)] = [{"properties": {"Station": "SFD 10"}}]
        app.session_state["station_area_map"] = {
            "selection": {"objects": dict(objects), "indices": {}}
        }
        app.run(timeout=60)
        stations = next(t for t in app.tabs if t.label == "Stations")
        self.assertIn("Comparing SFD 2, SFD 25", [s.value for s in stations.subheader])
        app.date_input[0].set_value(date(2026, 10, 31)).run(timeout=60)
        stations = next(t for t in app.tabs if t.label == "Stations")
        self.assertIn("Comparing SFD 2, SFD 25", [s.value for s in stations.subheader])

    def test_citywide_map_click_opens_history_once_and_picks_nothing(self):
        # Clicking SFD 10 on Explore > Citywide opens its history window; the next run (a new
        # layer id) does not reopen it, and no map or table pick changes.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        city = next(t for t in app.tabs if t.label == "Citywide")
        lid = self._pickable_id(city)
        self._tap_layer(app, "city_value_map", lid, ["SFD 10"])
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(windows(app), ["SFD 10: history known at planning time"])
        tab = {t.label: t for t in app.tabs}
        self.assertNotEqual(self._pickable_id(tab["Citywide"]), lid)
        self.assertEqual(app.multiselect[0].value, [])
        app.run(timeout=60)
        self.assertEqual(windows(app), [])
        # Every clickable map shades the hovered area the same way.
        tab = {t.label: t for t in app.tabs}
        for t in (tab["Prediction"], tab["Citywide"], tab["Stations"]):
            spec = json.loads(t.get("deck_gl_json_chart")[0].proto.json)
            layer = next(l for l in spec["layers"] if l.get("pickable"))
            self.assertEqual(layer["highlightColor"], HOVER_COLOR)

    def test_prediction_map_has_no_reset_button_and_keeps_its_key(self):
        # The map cannot move, so nothing resets it; clicks still reach the one fixed key.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertNotIn("Reset map view", [b.label for b in app.button])
        app.multiselect[0].select("SFD 10").select("SFD 2").run(timeout=60)
        self._tap_layer(app, "prediction_map", self._pickable_id(app.tabs[0]), ["SFD 25"])
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.multiselect[0].value, ["SFD 10", "SFD 2", "SFD 25"])

    def test_unpick_after_refused_click_and_date_change_never_restores_station(self):
        # Stale earlier layer keys stay in selection.objects; unpicking and changing date must hold.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for name in ["SFD 10", "SFD 2", "SFD 25", "SFD 5"]:
            click_area(app, name)
        self.assertEqual(app.multiselect[0].value, ["SFD 10", "SFD 2", "SFD 25"])
        click_area(app, "SFD 10")
        self.assertEqual(app.multiselect[0].value, ["SFD 2", "SFD 25"])
        app.date_input[0].set_value(date(2026, 10, 31)).run(timeout=60)
        self.assertEqual(app.multiselect[0].value, ["SFD 2", "SFD 25"])


if __name__ == "__main__":
    unittest.main()
