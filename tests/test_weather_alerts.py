"""Weather alerts on the Prediction tab: which forecast is used, the thresholds, and the wording.
Every forecast here is made up; nothing is downloaded.

Run: uv run python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import weather_alerts
from weather_alerts import MESSAGES, get_weather_alerts

TZ = "America/Los_Angeles"


def hourly(shifts, suffix=""):
    """Hourly UTC rows where every hour of each 08:00-08:00 shift has its (temperature, gust).
    `shifts` maps a shift date to (temperature, gust); DST shifts get 23 or 25 hours."""
    temps, gusts = {}, {}
    for day, (temp, gust) in shifts.items():
        start = (pd.Timestamp(day) + pd.Timedelta(hours=8)).tz_localize(TZ)
        end = (pd.Timestamp(day) + pd.Timedelta(days=1, hours=8)).tz_localize(TZ)
        for t in pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left"):
            t = t.tz_localize(None)
            temps[t] = temp
            gusts[t + pd.Timedelta(hours=1)] = gust  # gusts are labelled by the end of their hour
    frame = pd.DataFrame({f"temperature_2m{suffix}": temps, f"wind_gusts_10m{suffix}": gusts})
    return frame.rename_axis("time").reset_index()


def week(center, temps, gusts=None):
    """Seven shifts around `center` with the given lows (and gusts, default calm)."""
    days = pd.date_range(pd.Timestamp(center) - pd.Timedelta(days=3), periods=7)
    gusts = gusts or [20.0] * 7
    return {d: (t, g) for d, t, g in zip(days, temps, gusts)}


class WeatherAlertTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        (self.root / "data").mkdir()
        (self.root / "data/sfd_stations.csv").write_text(
            "station,address,lon,lat\nSFD 2,x,-122.34,47.61\nSFD 10,y,-122.33,47.60\n"
        )
        # no test may reach the network
        patcher = mock.patch.object(weather_alerts, "_download", side_effect=AssertionError("network"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def store(self, shifts):
        hourly(shifts, "_previous_day3").to_csv(self.root / weather_alerts.STORED_PATH, index=False)

    def alerts(self, day, now, fetch=None):
        return get_weather_alerts(day, pd.Timestamp(now), self.root, fetch)


class StoredForecastTests(WeatherAlertTestCase):
    # Demo dates: the day-3 forecast saved by step8_weather.py, no download.
    def test_strong_wind_at_the_threshold(self):
        self.store(week("2025-03-12", [5.0] * 7, [20, 20, 20, 65.0, 20, 20, 20]))
        out = self.alerts("2025-03-12", "2025-03-20")
        self.assertEqual(out["status"], "alert")
        self.assertEqual(out["source"], "stored")
        self.assertEqual(out["messages"], [MESSAGES["wind"]])

    def test_wind_just_below_the_threshold(self):
        self.store(week("2025-03-12", [5.0] * 7, [20, 20, 20, 64.9, 20, 20, 20]))
        self.assertEqual(self.alerts("2025-03-12", "2025-03-20")["status"], "clear")

    def test_hard_freeze_at_the_threshold(self):
        # neighbours are mild, so this is a hard freeze but not a cold spell
        self.store(week("2025-02-11", [5.0, 5.0, 5.0, -3.0, 5.0, 5.0, 5.0]))
        out = self.alerts("2025-02-11", "2025-02-20")
        self.assertEqual(out["messages"], [MESSAGES["freeze"]])

    def test_cold_but_not_hard_freeze_alone_is_clear(self):
        self.store(week("2025-02-11", [5.0, 5.0, 5.0, -2.9, 5.0, 5.0, 5.0]))
        out = self.alerts("2025-02-11", "2025-02-20")
        self.assertEqual(out["status"], "clear")
        self.assertEqual(out["messages"], [MESSAGES["clear"]])

    def test_cold_spell_uses_the_next_shift(self):
        self.store(week("2025-01-20", [5.0, 5.0, 5.0, -1.0, -1.5, 5.0, 5.0]))
        self.assertEqual(self.alerts("2025-01-20", "2025-01-30")["messages"], [MESSAGES["spell"]])

    def test_cold_spell_uses_the_previous_shift(self):
        self.store(week("2025-01-20", [5.0, 5.0, -1.2, -1.0, 5.0, 5.0, 5.0]))
        self.assertEqual(self.alerts("2025-01-20", "2025-01-30")["messages"], [MESSAGES["spell"]])

    def test_one_cold_shift_is_not_a_spell(self):
        self.store(week("2025-01-20", [5.0, 5.0, -0.9, -1.0, -0.9, 5.0, 5.0]))
        self.assertEqual(self.alerts("2025-01-20", "2025-01-30")["status"], "clear")

    def test_hard_freeze_in_a_spell_and_wind_give_two_lines(self):
        self.store(week("2025-02-11", [5.0, 5.0, -2.0, -4.0, 5.0, 5.0, 5.0], [20, 20, 20, 70, 20, 20, 20]))
        out = self.alerts("2025-02-11", "2025-02-20")
        self.assertEqual(out["messages"], [MESSAGES["wind"], MESSAGES["freeze_spell"]])

    def test_gust_in_the_last_hour_belongs_to_this_shift(self):
        # the gust for 07:00-08:00 is stamped 08:00, the start of the next shift
        shifts = week("2025-03-12", [5.0] * 7)
        self.store(shifts)
        path = self.root / weather_alerts.STORED_PATH
        table = pd.read_csv(path, parse_dates=["time"])
        end = pd.Timestamp("2025-03-13 08:00").tz_localize(TZ).tz_convert("UTC").tz_localize(None)
        table.loc[table["time"] == end, "wind_gusts_10m_previous_day3"] = 80.0
        table.to_csv(path, index=False)
        self.assertEqual(self.alerts("2025-03-12", "2025-03-20")["status"], "alert")
        self.assertEqual(self.alerts("2025-03-13", "2025-03-20")["status"], "clear")

    def test_25_hour_shift(self):
        # 2025-11-01 08:00 to 11-02 08:00 has 25 hours (clocks go back); its last hour,
        # 07:00 PST = 15:00 UTC, still belongs to it and not to the next shift
        self.store(week("2025-11-01", [5.0] * 7))
        path = self.root / weather_alerts.STORED_PATH
        table = pd.read_csv(path, parse_dates=["time"])
        table.loc[table["time"] == pd.Timestamp("2025-11-02 15:00"), "temperature_2m_previous_day3"] = -5.0
        table.to_csv(path, index=False)
        out = self.alerts("2025-11-01", "2025-11-10")
        self.assertEqual(out["source"], "stored")
        self.assertEqual(out["messages"], [MESSAGES["freeze"]])
        self.assertEqual(self.alerts("2025-11-02", "2025-11-10")["status"], "clear")


class ForecastSourceTests(WeatherAlertTestCase):
    def setUp(self):
        super().setUp()
        self.store(week("2025-03-12", [5.0] * 7))  # stored forecasts end before the dates below
        self.calls = []

    def fake(self, shifts):
        def fetch(url, params):
            self.calls.append((url, dict(params)))
            suffix = "_previous_day3" if "previous" in url else ""
            table = hourly(shifts, suffix)
            table["time"] = table["time"].dt.strftime("%Y-%m-%dT%H:%M")
            return {"hourly": table.to_dict(orient="list")}
        return fetch

    def test_planning_time_passed_uses_the_archived_day3_forecast(self):
        # Saturday's shift, now is Thursday 09:00: the planning time (Thursday 08:00) has passed
        fetch = self.fake(week("2025-06-07", [12.0] * 7, [20, 20, 20, 70, 20, 20, 20]))
        out = self.alerts("2025-06-07", "2025-06-05 09:00", fetch)
        self.assertEqual((out["status"], out["source"]), ("alert", "previous_runs"))
        url, params = self.calls[0]
        self.assertEqual(url, weather_alerts.PREVIOUS_RUNS_URL)
        self.assertEqual(params["models"], "gfs_global")
        self.assertIn("wind_gusts_10m_previous_day3", params["hourly"])

    def test_within_4_days_uses_the_live_forecast(self):
        # now is Wednesday 10:00: Saturday's shift ends Sunday 08:00, within 4 days
        fetch = self.fake(week("2025-06-07", [12.0] * 7))
        out = self.alerts("2025-06-07", "2025-06-04 10:00", fetch)
        self.assertEqual((out["status"], out["source"]), ("clear", "live"))
        url, params = self.calls[0]
        self.assertEqual(url, weather_alerts.FORECAST_URL)
        self.assertEqual(params["models"], "gfs_seamless")
        self.assertEqual(params["timezone"], "GMT")

    def test_more_than_4_days_ahead_has_no_forecast_yet(self):
        fetch = self.fake(week("2025-06-07", [12.0] * 7))
        out = self.alerts("2025-06-07", "2025-06-03 07:00", fetch)
        self.assertEqual(out, {"status": "not_yet", "messages": [MESSAGES["not_yet"]], "source": None})
        self.assertEqual(self.calls, [])

    def test_download_error_is_unavailable(self):
        def broken(url, params):
            raise weather_alerts.requests.ConnectionError("offline")
        out = self.alerts("2025-06-07", "2025-06-05 09:00", broken)
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(out["messages"], [MESSAGES["unavailable"]])

    def test_missing_hours_are_unavailable(self):
        def partial(url, params):
            table = self.fake(week("2025-06-07", [12.0] * 7))(url, params)
            table["hourly"]["temperature_2m_previous_day3"][80] = None
            return table
        self.assertEqual(self.alerts("2025-06-07", "2025-06-05 09:00", partial)["status"], "unavailable")


class WordingTests(unittest.TestCase):
    def test_messages_are_plain_and_have_no_numbers(self):
        for key, text in MESSAGES.items():
            with self.subTest(message=key):
                self.assertNotIn("%", text)
                for word in ["model", "MAE", "GFS", "ERA5", "km/h", "°C"]:
                    self.assertNotIn(word, text)
        self.assertNotIn("%", weather_alerts.NOTE)


if __name__ == "__main__":
    unittest.main()
