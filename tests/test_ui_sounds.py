"""The UI sounds: the three sound files, how Streamlit serves them, the one hidden script the
page mounts and the list of things that click. The browser side (what plays when) is checked
in headless Chromium.

Run: uv run python -m unittest discover -s tests -v
"""

import hashlib
import json
import os
import tomllib
import unittest
import wave

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
os.environ.setdefault("CALLCAST_OFFLINE", "1")  # also in tests/__init__.py

from streamlit.testing.v1 import AppTest  # noqa: E402

from app_data import ROOT  # noqa: E402
from ui_sounds import CLICKS, QUIET, SOUND_DIR, SOUNDS, SOUNDS_KEY  # noqa: E402


class SoundFileTests(unittest.TestCase):
    def test_each_sound_is_a_short_quiet_size_wav(self):
        # hover.wav is 0.015 s, click.wav 0.035 s and close.wav 0.06 s, all under 6 KB.
        for name in SOUNDS:
            path = SOUND_DIR / f"{name}.wav"
            self.assertLess(path.stat().st_size, 6000, name)
            with wave.open(str(path)) as w:
                self.assertLess(w.getnframes() / w.getframerate(), 0.1, name)

    def test_volumes_stay_low(self):
        self.assertEqual(set(SOUNDS), {"hover", "click", "close"})
        self.assertTrue(all(0 < v <= 0.5 for v in SOUNDS.values()))

    def test_close_is_the_studio_click_out(self):
        # close.wav is a copy of the studio's ui_back_dismiss.wav (SHA-1 36178b38...).
        digest = hashlib.sha1((SOUND_DIR / "close.wav").read_bytes()).hexdigest()
        self.assertEqual(digest, "36178b38b6720f77807673cd871c319d7c8f58c9")

    def test_streamlit_serves_the_static_folder(self):
        config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
        self.assertTrue(config["server"]["enableStaticServing"])
        self.assertEqual(SOUND_DIR, ROOT / "static" / "sounds")

    def test_docker_image_gets_the_sounds(self):
        self.assertIn("COPY --chown=appuser:appuser static static", (ROOT / "Dockerfile").read_text())
        ignored = (ROOT / ".dockerignore").read_text().splitlines()
        self.assertFalse([line for line in ignored if line.startswith("static")])


class SoundScriptTests(unittest.TestCase):
    def test_page_mounts_one_hidden_sound_script(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception)
        data = [json.loads(c.proto.json) for c in app.get("bidi_component")]
        sounds = [d for d in data if "volume" in d]
        want = {"url": "app/static/sounds/", "volume": SOUNDS, "clicks": CLICKS, "quiet": QUIET}
        self.assertEqual(sounds, [want])
        # style.css hides its box, so the page layout does not move.
        self.assertIn(f".st-key-{SOUNDS_KEY} {{\n  display: none", (ROOT / "style.css").read_text())

    def test_only_buttons_tabs_and_table_cells_click(self):
        # Clicks: buttons, tabs, button links and the table's tick, Expected and bars cells.
        self.assertEqual(CLICKS, ["button", "a[data-testid*='LinkButton']", "[role='tab']", "[data-act]"])
        html = (ROOT / "prediction_cards.py").read_text()
        for act in ("tick", "hourly", "history"):
            self.assertIn(f'data-act="{act}"', html)
        # Boxes and lists stay silent: no option, checkbox or combobox role is on the list.
        for role in ("option", "checkbox", "combobox", "select"):
            self.assertFalse([c for c in CLICKS if role in c], role)
        for box in ("stDateInput", "stMultiSelect", "stSelectbox", "listbox", "input"):
            self.assertTrue([q for q in QUIET if box in q], box)


if __name__ == "__main__":
    unittest.main()
