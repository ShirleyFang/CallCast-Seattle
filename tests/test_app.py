"""App behaviour tests: planning time, the historical average, model checks,
the hourly breakdown, and the Streamlit screen.

Run: uv run python -m unittest discover -s tests -v
"""

import functools
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app_data import (
    ROOT,
    first_model_date,
    forecast_origin,
    hourly_allocation,
    known_history,
    learned_predictions,
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
        # Forecast shifts after October 28 can be picked; October 28 stays the last day of counts.
        last_forecast = self.features.ShiftDate.max().date()
        self.assertEqual(last_forecast, date(2026, 10, 31))
        self.assertEqual(
            target_dates(self.counts, first_model_date(), last_forecast),
            (date(2026, 1, 3), date(2026, 10, 31), date(2026, 10, 24)),
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


# Forecast shifts: the three shifts after the last day of counts.


class ForecastTests(AppTestCase):
    def test_forecast_rows_are_added_without_calls(self):
        last_count = self.counts.ShiftDate.max()
        forecast = self.features[self.features.ShiftDate > last_count]
        self.assertEqual(len(forecast), 34 * 3)
        self.assertEqual(
            sorted(forecast.ShiftDate.dt.date.unique()),
            [date(2026, 10, 29), date(2026, 10, 30), date(2026, 10, 31)],
        )
        self.assertTrue(forecast.calls.isna().all())
        self.assertEqual(set(forecast.Station), set(self.ids))
        # Every shift up to the last count day still has its real calls.
        self.assertTrue(
            self.features[self.features.ShiftDate <= last_count].calls.notna().all()
        )

    def test_model_gives_34_stations_and_the_expected_totals(self):
        for target, total in [
            ("2026-10-29", 317.7),
            ("2026-10-30", 327.7),
            ("2026-10-31", 311.2),
        ]:
            result, _ = learned_predictions(self.features, target, self.ids)
            self.assertEqual(len(result), 34)
            self.assertEqual(round(result.predicted_calls.sum(), 1), total)

    def test_usual_number_comes_from_the_counts(self):
        # October 31 is a Saturday; its usual number averages earlier Saturdays from the counts.
        usual = reference_predictions(self.counts, "2026-10-31", self.ids)
        self.assertEqual(len(usual), 34)


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
    def test_default_screen_and_station_switch(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(len(app.sidebar.selectbox[0].options), 34)
        # No prediction source switch: the app always shows the trained model.
        self.assertEqual(len(app.sidebar.radio), 0)
        # Default view: Oct 24, SFD 10, the model's numbers.
        self.assertEqual(app.metric[0].value, "313.6")
        self.assertEqual(app.metric[1].value, "39.3")
        table = app.dataframe[0].value
        self.assertAlmostEqual(table["Expected calls"].sum(), 313.6, delta=0.05)
        self.assertEqual(
            list(table.columns),
            ["Station", "Expected calls", "Usual for this weekday", "Difference"],
        )
        # Switching station changes the station number, not the citywide total.
        app.sidebar.selectbox[0].select("SFD 2").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "313.6")
        self.assertEqual(app.metric[1].value, "32.1")

    def test_screen_shows_no_technical_details(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        text = " ".join(x.value for x in [*app.caption, *app.markdown, *app.info])
        for word in ["Poisson", "GBM", "first pass", "Trained model", "reference mode"]:
            self.assertNotIn(word, text)
        self.assertEqual(len(app.code), 0)
        # The two planner links are there, and the repository link is not.
        self.assertIn("sfdlive.com", text)
        self.assertIn("seattlebreaking.com", text)
        self.assertNotIn("github.com", text)

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
            (picker.min, picker.max), (date(2026, 1, 3), date(2026, 10, 31))
        )

    def test_forecast_dates_are_labelled_and_checked_on_screen(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for day, total in [
            (date(2026, 10, 29), "317.7"),
            (date(2026, 10, 30), "327.7"),
            (date(2026, 10, 31), "311.2"),
        ]:
            app.sidebar.date_input[0].set_value(day).run(timeout=60)
            self.assertFalse(app.exception, str(day))
            self.assertEqual(app.metric[0].value, total)
            self.assertEqual(len(app.dataframe[0].value), 34)
            self.assertIn("Forecast", app.subheader[0].value)
            self.assertTrue(any("Forecast:" in c.value for c in app.caption))
        # October 31 is 25 hours long, so the hourly view says so instead of drawing.
        notes = " ".join(x.value for e in app.expander for x in e.info)
        self.assertIn("25 hours long", notes)
        # The last day with real counts is not labelled as a forecast.
        app.sidebar.date_input[0].set_value(date(2026, 10, 28)).run(timeout=60)
        self.assertNotIn("Forecast", app.subheader[0].value)
        self.assertFalse(any("Forecast:" in c.value for c in app.caption))


if __name__ == "__main__":
    unittest.main()
