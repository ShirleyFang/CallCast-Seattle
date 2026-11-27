"""The weather parts of the app: the alert under the estimate on the Prediction tab, and the
Explore > Weather sub-tab. Nothing is downloaded (CALLCAST_OFFLINE).

Run: uv run python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
os.environ.setdefault("CALLCAST_OFFLINE", "1")  # also in tests/__init__.py

from streamlit.testing.v1 import AppTest  # noqa: E402

import weather_alerts  # noqa: E402
import weather_now  # noqa: E402
from app_data import ROOT  # noqa: E402
from weather_alerts import MESSAGES, NOTE, NWS_NOTE, NWS_URL  # noqa: E402
from text_blocks import note, plain  # noqa: E402

# 2026-03-11: the saved day-3 forecast has gusts of 65 km/h or more, so it has a wind alert
WINDY = date(2026, 3, 11)
# tomorrow's shift is always within the 4 days that get the NWS note
TOMORROW = date.today() + timedelta(days=1)
# The NWS note as it reads on screen, with its link shown as a chip: "... Service: NWS Seattle."
NWS_SHOWN = plain(note(NWS_NOTE))


def run_app(day):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    app.date_input[0].set_value(day).run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]
    return app


def numbers(app):
    # Everything the estimate shows: the cards, the call type mix and the station table.
    cards = [(m.label, m.value, m.delta) for m in app.metric]
    tables = [d.value.to_csv(index=False) for d in app.dataframe]
    return cards, tables


def notes(app):
    # The words of every text block on the page, one entry per block.
    return [plain(h.value) for h in app.get("html")]


class WeatherAlertOnScreenTests(unittest.TestCase):
    def test_wind_alert_is_shown_like_event_warnings(self):
        app = run_app(WINDY)
        self.assertIn(MESSAGES["wind"], [w.value for w in app.warning])
        self.assertIn(NOTE, notes(app))

    def test_alert_does_not_change_the_numbers(self):
        with_alert = numbers(run_app(WINDY))
        clear = {"status": "clear", "messages": [MESSAGES["clear"]], "source": "stored"}
        with mock.patch.object(weather_alerts, "get_weather_alerts", return_value=clear):
            app = run_app(WINDY)
        self.assertNotIn(MESSAGES["wind"], [w.value for w in app.warning])
        self.assertIn(MESSAGES["clear"], notes(app))
        self.assertEqual(numbers(app), with_alert)

    def test_a_date_without_a_saved_forecast_is_quiet_offline(self):
        # Oct 31: after the saved forecasts, so it would need a download
        app = run_app(date(2026, 10, 31))
        self.assertIn(MESSAGES["unavailable"], notes(app))
        self.assertNotIn(MESSAGES["wind"], [w.value for w in app.warning])

    def test_nws_note_under_an_alert(self):
        wind = {"status": "alert", "messages": [MESSAGES["wind"]], "kinds": ["wind"], "source": "live"}
        with mock.patch.object(weather_alerts, "get_weather_alerts", return_value=wind):
            app = run_app(TOMORROW)
        shown = notes(app)
        self.assertIn(MESSAGES["wind"], [w.value for w in app.warning])
        self.assertIn(NWS_SHOWN, shown)
        self.assertGreater(shown.index(NWS_SHOWN), shown.index(NOTE))

    def test_nws_note_without_an_alert(self):
        # offline, so our forecast is unavailable; the NWS note still shows
        app = run_app(TOMORROW)
        self.assertIn(MESSAGES["unavailable"], notes(app))
        self.assertIn(NWS_SHOWN, notes(app))

    # The NWS logo sits in front of the NWS note; the note's words and link stay the same.
    @unittest.skipUnless((weather_now.ICON_DIR / weather_now.NWS_LOGO).exists(), "no local NWS logo")
    def test_nws_note_shows_the_nws_logo(self):
        app = run_app(TOMORROW)
        block = [h.value for h in app.get("html") if "National Weather Service" in h.value][0]
        self.assertIn('src="app/static/weather_icons/nws.png" alt="National Weather Service"', block)
        self.assertIn(NWS_URL, block)
        self.assertEqual(plain(block), NWS_SHOWN)

    # Logo file missing (as in a fresh clone): the note is exactly as before, no image, no error.
    def test_missing_nws_logo_leaves_the_note_as_it_was(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(weather_now, "ICON_DIR", Path(tmp) / "weather_icons"):
                app = run_app(TOMORROW)
        blocks = [h.value for h in app.get("html") if "National Weather Service" in h.value]
        self.assertEqual(blocks, [note(NWS_NOTE)])
        self.assertFalse([h.value for h in app.get("html") if "st-key-prediction_nws" in h.value])

    def test_no_nws_note_for_a_past_shift(self):
        app = run_app(WINDY)
        self.assertNotIn(NWS_SHOWN, notes(app))

    def test_weather_captions_have_no_percentages(self):
        app = run_app(TOMORROW)
        weather = {NOTE, NWS_SHOWN, *MESSAGES.values()}
        shown = [t for t in notes(app) if t in weather]
        self.assertIn(NWS_SHOWN, shown)
        for text in shown:
            self.assertNotIn("%", text)

    def test_alert_sits_above_the_event_warning_on_an_event_day(self):
        # Oct 25 is a Seahawks game: the citywide wind alert sits under the briefing, the
        # SFD 10 event warning in its compare card below, each with its own note.
        wind = {"status": "alert", "messages": [MESSAGES["wind"]], "kinds": ["wind"], "source": "stored"}
        with mock.patch.object(weather_alerts, "get_weather_alerts", return_value=wind):
            app = run_app(date(2026, 10, 25))
            app.multiselect[0].select("SFD 10").run(timeout=60)
        warnings = [w.value for w in app.warning]
        self.assertEqual(warnings[0], MESSAGES["wind"])
        self.assertIn("Seahawks home game", warnings[1])
        self.assertIn(NOTE, notes(app))
        event_note = (
            "Event warnings are informational and are not included in the forecast."
        )
        self.assertIn(event_note, notes(app))

    def test_weather_line_stays_on_a_date_without_an_estimate(self):
        # Jan 9 has no estimate (a gap in the data), but the saved forecast still gives a weather line.
        app = run_app(date(2026, 1, 9))
        self.assertIn(MESSAGES["clear"], notes(app))


# The <style> blocks that put an icon beside the weather alerts.
def alert_styles(app):
    return [h.value for h in app.get("html") if "st-key-weather_alert_" in h.value]


class AlertIconOnScreenTests(unittest.TestCase):
    # Each alert kind shows its icon; the freeze and spell message shows the snowflake.
    @unittest.skipUnless(weather_now.ICON_DIR.exists(), "no local icon folder")
    def test_each_alert_shows_its_icon(self):
        icons = {"wind": "alert_wind", "freeze": "alert_freeze", "spell": "alert_spell",
                 "freeze_spell": "alert_freeze"}
        for kind, icon in icons.items():
            with self.subTest(kind=kind):
                out = {"status": "alert", "messages": [MESSAGES[kind]], "kinds": [kind], "source": "stored"}
                with mock.patch.object(weather_alerts, "get_weather_alerts", return_value=out):
                    app = run_app(WINDY)
                self.assertEqual([w.value for w in app.warning], [MESSAGES[kind]])
                styles = alert_styles(app)
                self.assertEqual(len(styles), 1)
                self.assertIn(f".st-key-weather_alert_{kind} ", styles[0])
                self.assertIn(f"weather_icons/{icon}.svg", styles[0])

    # Icon folder moved away (as in a fresh clone): same alert words, no icon style, no error.
    def test_missing_icon_folder_leaves_the_alerts_as_they_were(self):
        both = {"status": "alert", "messages": [MESSAGES["wind"], MESSAGES["freeze_spell"]],
                "kinds": ["wind", "freeze_spell"], "source": "stored"}
        with tempfile.TemporaryDirectory() as tmp:
            with (mock.patch.object(weather_now, "ICON_DIR", os.path.join(tmp, "weather_icons")),
                  mock.patch.object(weather_alerts, "get_weather_alerts", return_value=both)):
                app = run_app(WINDY)
        self.assertEqual([w.value for w in app.warning], both["messages"])
        self.assertEqual(alert_styles(app), [])


class WeatherSubTabTests(unittest.TestCase):
    def test_weather_sub_tab_shows_three_charts_with_captions(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("Weather", [t.label for t in app.tabs])
        headings = [m.value for m in app.markdown]
        for heading in ["##### Calls on cold and hot days", "##### Calls during a cold spell",
                        "##### Past weather events"]:
            self.assertIn(heading, headings)
        weather = next(t for t in app.tabs if t.label == "Weather")
        captions = " ".join(plain(h.value) for h in weather.get("html"))
        self.assertIn("Calls kept rising the longer a cold spell lasted", captions)
        self.assertIn("The shaded days are the event", captions)
        for word in ["MAE", "model", "ERA5"]:
            self.assertNotIn(word, captions)

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
