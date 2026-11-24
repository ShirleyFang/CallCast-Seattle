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

from streamlit.testing.v1 import AppTest

# AppTest runs Streamlit outside its normal server context; suppress only its
# expected missing-context messages while keeping other warnings visible.


class _IgnoreBareModeWarning(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext" not in record.getMessage()


logging.getLogger(
    "streamlit.runtime.scriptrunner_utils.script_run_context"
).addFilter(_IgnoreBareModeWarning())

from app_data import (
    ROOT,
    add_likely_range,
    citywide_range,
    first_model_date,
    forecast_origin,
    hourly_allocation,
    known_history,
    learned_predictions,
    load_ranges,
    load_tables,
    reference_predictions,
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


def write_profile(root, history_through="2024-12-31"):
    # Even 1/24 share per hour for SFD 10, saved where the app looks for it.
    (root / "data").mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "Station": ["SFD 10"] * 24,
            "shift_hour": range(24),
            "share": [1 / 24] * 24,
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
        # Real data: first pick is a month after Jan 1 2022, default is Oct 24.
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
        self.assertEqual(
            target_dates(self.counts, first_model_date(), last_forecast),
            (date(2026, 1, 3), date(2027, 1, 26), date(2026, 10, 24)),
        )
        self.assertEqual(self.counts.ShiftDate.max().date(), date(2026, 10, 28))

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
        # 1.0 would go below zero, 5.0 is the top of the first band, 39.3 is SFD 10 on Oct 24.
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


class ScreenTests(unittest.TestCase):
    def test_tab_names_and_explore_charts(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        # Explore has three sub-tabs (Citywide, Stations, Events); AppTest lists them in page order.
        self.assertEqual(
            [t.label for t in app.tabs],
            ["Prediction", "Explore", "Citywide", "Stations", "Events", "Data and limits"],
        )
        # Explore keeps the history charts; model performance stays out of the app.
        explore = " ".join(c.value for c in app.tabs[1].caption)
        self.assertIn("Same history as the Prediction tab", explore)
        shown = " ".join(
            x.value
            for tab in app.tabs
            for x in (*tab.markdown, *tab.caption, *tab.subheader)
        )
        self.assertNotIn("station_errors", shown)
        # Our word for a gap in the feed stays off the screen.
        self.assertNotIn("outage", shown)
        self.assertIn("missing data", shown)

    def test_default_screen_and_station_switch(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(len(app.sidebar.selectbox[0].options), 34)
        # No prediction source switch: the app always shows the trained model.
        self.assertEqual(len(app.sidebar.radio), 0)
        # Default view: Oct 24, SFD 10, the model's numbers.
        self.assertEqual(app.metric[0].value, "312.8")
        self.assertEqual(app.metric[1].value, "40.0")
        # Both cards compare with the usual number: 312.8 - 306.4 = +6.3 (unrounded sums).
        self.assertEqual(app.metric[0].delta, "+6.3 vs usual")
        self.assertEqual(app.metric[1].delta, "+1.1 vs usual")
        table = app.dataframe[0].value
        self.assertAlmostEqual(table["Expected calls"].sum(), 312.8, delta=0.05)
        # The table covers all 34 stations, none below 0, and the city card is their sum.
        self.assertEqual(len(table), 34)
        self.assertTrue((table["Expected calls"] >= 0).all())
        self.assertAlmostEqual(
            float(app.metric[0].value.replace(",", "")),
            table["Expected calls"].sum(),
            delta=0.05,
        )
        self.assertEqual(
            list(table.columns),
            [
                "Station",
                "Expected calls",
                "Low",
                "High",
                "Usual for this weekday",
                "Difference",
            ],
        )
        self.assertTrue((table["Low"] <= table["Expected calls"]).all())
        self.assertTrue((table["Expected calls"] <= table["High"]).all())
        # Switching station changes the station number, not the citywide total.
        app.sidebar.selectbox[0].select("SFD 2").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "312.8")
        self.assertEqual(app.metric[1].value, "31.7")

    def test_calendar_days_are_readable_on_the_dark_sidebar(self):
        # The picker opens on the sidebar's secondary color, so the days you can pick
        # need light text; the default text color gave a contrast of 1.26 to 1.
        theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
        background = theme["theme"]["sidebar"]["secondaryBackgroundColor"]
        css = (ROOT / "style.css").read_text()
        rule = re.search(
            r'\[data-testid="stDateInputCalendar"\][^{]*:not\(\[aria-disabled="true"\]\)'
            r"\s*\{\s*color:\s*(#[0-9A-Fa-f]{6});",
            css,
        )
        self.assertIsNotNone(rule, "style.css lost the calendar day rule")

        def luminance(color):
            channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            r, g, b = [
                c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in channels
            ]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        light, dark = sorted([luminance(rule.group(1)), luminance(background)])[::-1]
        self.assertGreaterEqual((light + 0.05) / (dark + 0.05), 4.5)

    def test_screen_shows_no_technical_details(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        text = " ".join(
            x.value for x in [*app.caption, *app.markdown, *app.info, *app.warning]
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
        app.sidebar.date_input[0].set_value(date(2026, 1, 9)).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(
            [w.value for w in app.warning],
            ["No estimate for this date: the call data for this shift has a gap."],
        )
        self.assertEqual(len(app.metric), 0)
        shown = " ".join(x.value for x in [*app.warning, *app.info, *app.caption])
        self.assertNotIn(".csv", shown)

    def test_explore_charts_use_plain_names_in_order(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        # The only bar chart drawn by Streamlit itself is the hourly one on the Prediction tab.
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
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
        self.assertEqual(list(weekday["data"][0]["x"]), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        category = next(f for f in bars if f["data"][0].get("orientation") == "h")
        self.assertEqual([t["name"] for t in category["data"]], ["Fire", "Medical", "Non-emergency", "Other"])

    def test_likely_range_is_shown_for_city_station_and_table(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        captions = [c.value for c in app.caption]
        # The city and the selected station (SFD 10, expected 40.0) each get a range.
        self.assertIn("Likely range: 283 to 345", captions)
        self.assertIn("Likely range: 31 to 50", captions)
        meaning = next(c for c in captions if "4 in 5" in c)
        self.assertIn("not a staffing recommendation", meaning)
        self.assertIn("Low and High show the likely range for each station.", captions)
        # No model scores or fold talk next to it.
        text = " ".join(captions + [x.value for x in app.markdown])
        for word in ["MAE", "RMSE", "fold", "cross-valid", "cross valid"]:
            self.assertNotIn(word, text)
        # The range follows the date: Oct 31 has its own estimate and range.
        app.sidebar.date_input[0].set_value(date(2026, 10, 31)).run(timeout=60)
        self.assertIn("Likely range: 282 to 344", [c.value for c in app.caption])

    def test_model_error_is_looked_up_and_shown_once(self):
        # A model failure should give one warning, and the page should ask the model once.
        with mock.patch(
            "app_data.learned_predictions", side_effect=ValueError("model broke")
        ) as lookup:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual([w.value for w in app.warning], ["model broke"])

    def test_date_picker_range_on_screen(self):
        # The picker starts where the model can serve and ends at the last forecast date.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        picker = app.sidebar.date_input[0]
        self.assertEqual(
            (picker.min, picker.max), (date(2026, 1, 3), date(2027, 1, 26))
        )
        # The sidebar tells the supervisor how far ahead estimates go.
        self.assertIn(
            "Estimates are available up to Jan 26, 2027.",
            [c.value for c in app.sidebar.caption],
        )

    def test_forecast_dates_are_labelled_and_checked_on_screen(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for day, total in [
            (date(2026, 10, 29), "317.5"),
            (date(2026, 10, 30), "329.5"),
            (date(2026, 10, 31), "311.6"),
        ]:
            app.sidebar.date_input[0].set_value(day).run(timeout=60)
            self.assertFalse(app.exception, str(day))
            self.assertEqual(app.metric[0].value, total)
            self.assertEqual(len(app.dataframe[0].value), 34)
            self.assertIn("Forecast", app.subheader[0].value)
            self.assertTrue(any("Forecast:" in c.value for c in app.caption))
            # The first three forecast shifts still have real recent calls.
            self.assertNotIn("long-range", app.subheader[0].value)
        # October 31 is 25 hours long, so the hourly view says so instead of drawing.
        notes = " ".join(x.value for e in app.expander for x in e.info)
        self.assertIn("25 hours long", notes)
        # The last day with real counts is not labelled as a forecast.
        app.sidebar.date_input[0].set_value(date(2026, 10, 28)).run(timeout=60)
        self.assertNotIn("Forecast", app.subheader[0].value)
        self.assertFalse(any("Forecast:" in c.value for c in app.caption))

    def test_explore_says_which_shifts_it_uses(self):
        # Explore follows the planning time; past the data, every date uses the same history.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)

        def explore_note_and_table():
            explore = next(t for t in app.tabs if t.label == "Explore")
            city = next(t for t in app.tabs if t.label == "Citywide")
            note = next(c.value for c in explore.caption if c.value.startswith("Using finished shifts"))
            return note, city.dataframe[0].value

        for day, expected in [
            # Jan 9 is an outage shift, so the last usable one before the planning time is Jan 8;
            # that is not the end of the data.
            (date(2026, 1, 12), "up to Thu Jan 08, 2026, the last shift finished by the planning time."),
            (date(2026, 10, 31), "up to Wed Oct 28, 2026, the last shift finished by the planning time."),
            (date(2026, 11, 1), "up to Wed Oct 28, 2026, the last day of data. Later dates show the same history"),
        ]:
            app.sidebar.date_input[0].set_value(day).run(timeout=60)
            self.assertFalse(app.exception, str(day))
            self.assertIn(expected, explore_note_and_table()[0])
        _, nov = explore_note_and_table()
        app.sidebar.date_input[0].set_value(date(2027, 1, 20)).run(timeout=60)
        pd.testing.assert_frame_equal(nov, explore_note_and_table()[1])

    def test_long_range_dates_are_labelled_on_screen(self):
        # From Nov 1 (4 shifts after the last counts) to Jan 26, 2027 the estimate uses
        # typical recent-call values, and the screen says so.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for day in [date(2026, 11, 1), date(2026, 12, 25), date(2027, 1, 26)]:
            app.sidebar.date_input[0].set_value(day).run(timeout=60)
            self.assertFalse(app.exception, str(day))
            self.assertIn("Forecast", app.subheader[0].value)
            self.assertIn("long-range estimate", app.subheader[0].value)
            self.assertTrue(
                any(c.value.startswith("Long-range estimate:") for c in app.caption)
            )
            self.assertEqual(len(app.dataframe[0].value), 34)
        # Every estimate says that weather and events are not included.
        self.assertTrue(any("does not include weather" in c.value for c in app.caption))


if __name__ == "__main__":
    unittest.main()
