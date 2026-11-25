"""The weather parts of the app: the alert under the estimate on the Prediction tab, and the
Explore > Weather sub-tab. Nothing is downloaded (CALLCAST_OFFLINE).

Run: uv run python -m unittest discover -s tests -v
"""

import os
import unittest
from datetime import date
from unittest import mock

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
os.environ.setdefault("CALLCAST_OFFLINE", "1")  # also in tests/__init__.py

from streamlit.testing.v1 import AppTest  # noqa: E402

import weather_alerts  # noqa: E402
from app_data import ROOT  # noqa: E402
from weather_alerts import MESSAGES, NOTE  # noqa: E402

# 2026-03-11: the saved day-3 forecast has gusts of 65 km/h or more, so it has a wind alert
WINDY = date(2026, 3, 11)


def run_app(day):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    app.sidebar.date_input[0].set_value(day).run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]
    return app


def numbers(app):
    """Everything the estimate shows: the two cards and the station table."""
    cards = [(m.label, m.value, m.delta) for m in app.metric]
    table = app.dataframe[0].value.to_csv(index=False)
    return cards, table


class WeatherAlertOnScreenTests(unittest.TestCase):
    def test_wind_alert_is_shown_like_event_warnings(self):
        app = run_app(WINDY)
        self.assertIn(MESSAGES["wind"], [w.value for w in app.warning])
        self.assertIn(NOTE, [c.value for c in app.caption])

    def test_alert_does_not_change_the_numbers(self):
        with_alert = numbers(run_app(WINDY))
        clear = {"status": "clear", "messages": [MESSAGES["clear"]], "source": "stored"}
        with mock.patch.object(weather_alerts, "get_weather_alerts", return_value=clear):
            app = run_app(WINDY)
        self.assertNotIn(MESSAGES["wind"], [w.value for w in app.warning])
        self.assertIn(MESSAGES["clear"], [c.value for c in app.caption])
        self.assertEqual(numbers(app), with_alert)

    def test_a_date_without_a_saved_forecast_is_quiet_offline(self):
        # Oct 31: after the saved forecasts, so it would need a download
        app = run_app(date(2026, 10, 31))
        self.assertIn(MESSAGES["unavailable"], [c.value for c in app.caption])
        self.assertNotIn(MESSAGES["wind"], [w.value for w in app.warning])


class WeatherSubTabTests(unittest.TestCase):
    def test_weather_sub_tab_shows_three_charts_with_captions(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("Weather", [t.label for t in app.tabs])
        headings = [m.value for m in app.markdown]
        for heading in ["##### Calls on cold and hot days", "##### Calls during a cold spell",
                        "##### Past weather events"]:
            self.assertIn(heading, headings)
        captions = " ".join(c.value for c in app.caption)
        self.assertIn("Calls kept rising the longer a cold spell lasted", captions)
        self.assertIn("The shaded days are the event", captions)
        for word in ["MAE", "model", "ERA5"]:
            self.assertNotIn(word, " ".join(c.value for c in app.caption if "cold" in c.value
                                            or "event" in c.value))

    def test_picking_another_event(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        picker = app.get("button_group")
        picker = [p for p in picker if p.label == "Weather event"][0]
        picker.set_value("Jul 2022 heat wave").run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(any(m.value.startswith("**Jul 2022 heat wave**: Highest high")
                            for m in app.markdown))


if __name__ == "__main__":
    unittest.main()
