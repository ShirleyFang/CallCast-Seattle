"""The weather icon line under the briefing on the Prediction tab: code to icon, the shift summary,
which forecast it uses, and that it never changes the estimate. Nothing is downloaded.

Run: uv run python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
os.environ.setdefault("CALLCAST_OFFLINE", "1")  # also in tests/__init__.py

from streamlit.testing.v1 import AppTest  # noqa: E402

import weather_alerts  # noqa: E402
import weather_now  # noqa: E402
from app_data import ROOT  # noqa: E402
from weather_now import (  # noqa: E402
    ALERT_ICONS,
    CONDITIONS,
    ICON_DIR,
    code_shifts,
    get_shift_weather,
    notable_code,
    alert_icon_css,
    shift_condition,
    weather_now_html,
    weather_text,
)

TZ = "America/Los_Angeles"


# The UTC hours of the 08:00-08:00 shift on `day` (23 or 25 on DST days).
def shift_hours(day):
    start = (pd.Timestamp(day) + pd.Timedelta(hours=8)).tz_localize(TZ).tz_convert("UTC")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1, hours=8)).tz_localize(TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="h", inclusive="left").tz_localize(None)


def hourly(shifts, suffix=""):
    """Hourly UTC rows; `shifts` maps a date to (codes, cloud, temp, gust, rain mm/h, snow cm/h).
    Codes, rain, snow and gusts are labelled by the end of their hour, as Open-Meteo does."""
    rows = {}
    for day, (codes, cloud, temp, gust, rain, snow) in shifts.items():
        hours = shift_hours(day)
        codes = codes if isinstance(codes, list) else [codes] * len(hours)
        for t, code in zip(hours, codes):
            rows.setdefault(t, {}).update(temperature_2m=temp, cloud_cover=cloud)
            rows.setdefault(t + pd.Timedelta(hours=1), {}).update(
                weather_code=code, wind_gusts_10m=gust, precipitation=rain, snowfall=snow
            )
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return frame.add_suffix(suffix).rename_axis("time").reset_index()


def calm(day, code=3, cloud=50.0, temp=8.0, gust=20.0, rain=0.0, snow=0.0):
    return {pd.Timestamp(day) + pd.Timedelta(days=k): (code, cloud, temp, gust, rain, snow) for k in (-1, 0, 1)}


class ConditionTests(unittest.TestCase):
    def test_codes_give_their_icon(self):
        for code, condition in [(95, "thunder"), (71, "snow"), (66, "sleet"), (65, "heavy_rain"),
                                (61, "rain"), (51, "rain"), (45, "fog")]:
            self.assertEqual(shift_condition(code, 10.0, 20.0), condition)

    def test_dry_shifts_use_wind_then_cloud(self):
        # gusts of 65 km/h are the wind alert's threshold
        self.assertEqual(shift_condition(3, 90.0, 65.0), "wind")
        self.assertEqual(shift_condition(3, 90.0, 64.9), "cloudy")
        self.assertEqual(shift_condition(2, 50.0, 20.0), "partly_cloudy")
        self.assertEqual(shift_condition(0, 10.0, 20.0), "sun")

    def test_fog_and_rain_beat_wind(self):
        self.assertEqual(shift_condition(45, 100.0, 80.0), "fog")
        self.assertEqual(shift_condition(61, 100.0, 80.0), "rain")

    def test_one_hour_of_thunder_beats_a_day_of_rain(self):
        self.assertEqual(notable_code(pd.Series([61.0] * 23 + [95.0])), 95)
        self.assertEqual(notable_code(pd.Series([73.0, 65.0, 66.0])), 73)
        self.assertEqual(notable_code(pd.Series([0.0, 3.0, 1.0])), 3)

    # The icons are local files kept out of git, so a fresh clone skips this.
    @unittest.skipUnless(ICON_DIR.exists(), "no local icon folder")
    def test_every_icon_file_is_there(self):
        for condition in CONDITIONS:
            self.assertTrue((ICON_DIR / f"{condition}.svg").exists(), condition)


class AlertIconTests(unittest.TestCase):
    @unittest.skipUnless(ICON_DIR.exists(), "no local icon folder")
    def test_every_alert_icon_file_is_there(self):
        for name in set(ALERT_ICONS.values()):
            self.assertTrue((ICON_DIR / f"{name}.svg").exists(), name)

    # Wind gets the windsock, the freeze and spell message the hard freeze snowflake.
    def test_each_alert_kind_gets_its_icon(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ALERT_ICONS.values():
                (Path(tmp) / f"{name}.svg").write_text("<svg/>")
            css = alert_icon_css(["wind", "freeze_spell"], icon_dir=tmp)
            spell = alert_icon_css(["spell"], icon_dir=tmp)
        self.assertTrue(css.startswith("<style>"))
        self.assertIn('.st-key-weather_alert_wind [data-testid="stAlertContentWarning"]::before', css)
        self.assertIn("weather_icons/alert_wind.svg", css)
        self.assertIn(".st-key-weather_alert_freeze_spell ", css)
        self.assertIn("weather_icons/alert_freeze.svg", css)
        self.assertIn("weather_icons/alert_spell.svg", spell)

    def test_no_alert_icon_file_gives_no_style(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(alert_icon_css(["wind", "freeze"], icon_dir=empty), "")
        self.assertEqual(alert_icon_css([]), "")


class ShiftSummaryTests(unittest.TestCase):
    def test_code_shifts_takes_the_shift_hours(self):
        # Wed Mar 11: rain from the 4th hour; Mar 10 and Mar 12 are dry
        day = pd.Timestamp("2026-03-11")
        codes = [3] * 3 + [63] * (len(shift_hours(day)) - 3)
        table = hourly({**calm(day), day: (codes, 100.0, 6.0, 30.0, 1.0, 0.0)})
        out = code_shifts(table)
        self.assertEqual(out.loc[day, "weather_code"], 63)
        self.assertEqual(out.loc[day, "cloud_cover"], 100.0)
        self.assertEqual(out.loc[day - pd.Timedelta(days=1), "weather_code"], 3)

    def test_a_shift_with_a_missing_hour_is_left_out(self):
        day = pd.Timestamp("2026-03-11")
        table = hourly(calm(day))
        table.loc[30, "cloud_cover"] = None
        self.assertNotIn(day, code_shifts(table).index)

    def test_dst_shift_is_complete(self):
        # Sun Mar 8 2026 has 23 hours
        day = pd.Timestamp("2026-03-08")
        self.assertIn(day, code_shifts(hourly(calm(day))).index)

    def test_text_shows_rain_or_snow_only_above_zero(self):
        base = {"label": "Rain", "temp_min": -0.4, "temp_max": 9.2, "rain_mm": 12.34, "snow_cm": 0.0}
        self.assertEqual(weather_text(base), "Rain · Low 0 °C, high 9 °C · 12.3 mm rain")
        self.assertEqual(weather_text({**base, "label": "Cloudy", "rain_mm": 0.02}),
                         "Cloudy · Low 0 °C, high 9 °C")
        self.assertEqual(weather_text({**base, "label": "Snow", "rain_mm": 8.0, "snow_cm": 4.2}),
                         "Snow · Low 0 °C, high 9 °C · 4.2 cm snow")

    def test_html_has_the_icon_and_alt_text(self):
        summary = {"condition": "rain", "label": "Rain", "temp_min": 4.0, "temp_max": 11.0,
                   "rain_mm": 27.6, "snow_cm": 0.0}
        # a stand-in rain.svg, so the test does not need the local icon folder
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "rain.svg").write_text("<svg/>")
            html = weather_now_html(summary, icon_dir=tmp)
        self.assertIn('src="app/static/weather_icons/rain.svg"', html)
        self.assertIn('alt="Rain"', html)
        self.assertIn("27.6 mm rain", html)

    def test_no_summary_gives_nothing(self):
        self.assertEqual(weather_now_html(None), "")

    # A clone without the icon folder still gets the words, with no image tag.
    def test_no_icon_file_shows_the_words_only(self):
        summary = {"condition": "rain", "label": "Rain", "temp_min": 4.0, "temp_max": 11.0,
                   "rain_mm": 27.6, "snow_cm": 0.0}
        with tempfile.TemporaryDirectory() as empty:
            html = weather_now_html(summary, icon_dir=empty)
        self.assertNotIn("<img", html)
        self.assertIn("Rain</strong> · Low 4 °C, high 11 °C · 27.6 mm rain", html)


class ForecastSourceTests(unittest.TestCase):
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

    def store(self, shifts, codes=True):
        table = hourly(shifts, "_previous_day3")
        weather = ["time", *(f"{v}_previous_day3" for v in weather_now.SHIFT_VARIABLES)]
        table[weather].to_csv(self.root / weather_alerts.STORED_PATH, index=False)
        if codes:
            plain = hourly(shifts)
            code_shifts(plain).to_csv(self.root / weather_now.CODES_PATH, date_format="%Y-%m-%d")

    def fake(self, shifts):
        self.calls = []

        def fetch(url, params):
            self.calls.append((url, dict(params)))
            suffix = "_previous_day3" if url == weather_alerts.PREVIOUS_RUNS_URL else ""
            return {"hourly": hourly(shifts, suffix).to_dict("list")}
        return fetch

    def test_saved_demo_date(self):
        self.store(calm("2026-03-11", code=63, cloud=100.0, temp=6.0, rain=1.0))
        out = get_shift_weather("2026-03-11", pd.Timestamp("2026-11-27"), self.root)
        self.assertEqual(out["source"], "stored")
        self.assertEqual(out["condition"], "rain")
        self.assertAlmostEqual(out["rain_mm"], len(shift_hours("2026-03-11")))

    def test_saved_date_without_codes_shows_nothing(self):
        self.store(calm("2026-03-11"), codes=False)
        self.assertIsNone(get_shift_weather("2026-03-11", pd.Timestamp("2026-11-27"), self.root))

    def test_previous_runs_share_the_alerts_download(self):
        fetch = self.fake(calm("2025-06-07", code=0, cloud=10.0, temp=12.0))
        out = get_shift_weather("2025-06-07", pd.Timestamp("2025-06-20"), self.root, fetch)
        weather_alerts.get_weather_alerts("2025-06-07", pd.Timestamp("2025-06-20"), self.root, fetch)
        self.assertEqual(out["source"], "previous_runs")
        self.assertEqual(out["condition"], "sun")
        self.assertIn("weather_code_previous_day3", self.calls[0][1]["hourly"])
        # same url and params, so Streamlit's cache downloads it once for both
        self.assertEqual(self.calls[0], self.calls[1])

    def test_live_forecast_after_the_planning_time_is_ahead(self):
        fetch = self.fake(calm("2026-11-30", code=71, cloud=100.0, temp=-1.0, snow=0.2))
        out = get_shift_weather("2026-11-30", pd.Timestamp("2026-11-27 09:30"), self.root, fetch)
        self.assertEqual(out["source"], "live")
        self.assertEqual(out["condition"], "snow")
        self.assertEqual(self.calls[0][1]["models"], "gfs_seamless")

    def test_offline_gives_no_icon(self):
        self.assertIsNone(get_shift_weather("2026-11-30", pd.Timestamp("2026-11-27 09:30"), self.root))

    def test_too_far_ahead_gives_no_icon(self):
        fetch = self.fake(calm("2026-12-20"))
        self.assertIsNone(get_shift_weather("2026-12-20", pd.Timestamp("2026-11-27"), self.root, fetch))
        self.assertEqual(self.calls, [])


def run_app(day):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    app.date_input[0].set_value(day).run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]
    return app


def numbers(app):
    cards = [(m.label, m.value, m.delta) for m in app.metric]
    tables = [d.value.to_csv(index=False) for d in app.dataframe]
    return cards, tables


# The weather line, found by its words so it works with or without the icon files.
def lines(app):
    return [h.value for h in app.get("html") if " °C, high " in h.value]


class WeatherIconOnScreenTests(unittest.TestCase):
    @unittest.skipUnless(ICON_DIR.exists(), "no local icon folder")
    def test_demo_date_shows_its_icon(self):
        # Wed Mar 11 2026: the saved day-3 forecast has rain all shift
        shown = lines(run_app(date(2026, 3, 11)))
        self.assertEqual(len(shown), 1)
        self.assertIn('alt="Rain"', shown[0])

    def test_icon_does_not_change_the_estimate(self):
        with_icon = run_app(date(2026, 3, 11))
        with mock.patch.object(weather_now, "get_shift_weather", return_value=None):
            without = run_app(date(2026, 3, 11))
        self.assertEqual(lines(without), [])
        self.assertEqual(numbers(without), numbers(with_icon))

    def test_offline_date_shows_nothing(self):
        # Oct 31: after the saved forecasts, so it would need a download
        app = run_app(date(2026, 10, 31))
        self.assertEqual(lines(app), [])

    # The icon folder moved away (as in a fresh clone): Mar 11 shows the words, no image, no error.
    def test_missing_icon_folder_shows_the_words_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(weather_now, "ICON_DIR", Path(tmp) / "weather_icons"):
                shown = lines(run_app(date(2026, 3, 11)))
        self.assertEqual(len(shown), 1)
        self.assertNotIn("<img", shown[0])
        self.assertIn("Rain</strong> · Low", shown[0])
