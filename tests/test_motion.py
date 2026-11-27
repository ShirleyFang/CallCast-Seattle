"""Hover and press motion on clickable parts, the tab bars and their icons.

Run: uv run python -m unittest discover -s tests -v
"""

import re
import unittest

from app_data import ROOT
from prediction_cards import TABLE_CSS, TABLE_JS

CSS = (ROOT / "style.css").read_text()
ICONS = ROOT / "static" / "tab_icons"


class MotionTests(unittest.TestCase):
    def test_motion_settings_live_in_one_place(self):
        # Lift, press and timings are set once in a :root block of style.css; the table reads them.
        root = "".join(re.findall(r":root \{([^}]*)\}", CSS))
        for name in ["lift", "press", "press-wide", "hover-time", "press-time", "slide-time"]:
            self.assertIn(f"--cc-{name}:", root)
        for name in ["lift", "press", "hover-time", "press-time"]:
            self.assertIn(f"var(--cc-{name})", TABLE_CSS)

    def test_hover_only_with_a_mouse_and_off_for_reduced_motion(self):
        # Every lift sits inside @media (hover: hover), so phones get the press only.
        for css in (CSS, TABLE_CSS):
            lifts = [m.start() for m in re.finditer(r"translateY\(var\(--cc-lift\)\)", css)]
            self.assertTrue(lifts)
            for at in lifts:
                block = css[css.rindex("@media", 0, at):at]
                self.assertTrue(block.startswith("@media (hover: hover)"))
                self.assertNotIn("\n}", block)
            self.assertIn("prefers-reduced-motion: reduce", css)
        reduced = CSS[CSS.rindex("/* Reduced motion"):]
        self.assertIn("transform: none !important", reduced)

    def test_motion_moves_only_transform_opacity_or_filter(self):
        # No glow or growing shadow: transitions never name box-shadow.
        for css in (CSS, TABLE_CSS):
            for rule in re.findall(r"transition:([^;]*);", css):
                self.assertNotIn("box-shadow", rule)
                self.assertNotIn("all", rule.split())

    def test_table_press_starts_on_pointer_down(self):
        # The press class goes on at pointer down, before the click reruns the page.
        self.assertIn("onpointerdown", TABLE_JS)
        self.assertIn('classList.add("press")', TABLE_JS)
        self.assertIn("@keyframes press", TABLE_CSS)

    def test_every_icon_in_the_css_exists_and_matches_the_set(self):
        # 3 top tabs, 4 Explore views, 3 shares and 5 call types, all on one 24 px grid.
        used = set(re.findall(r"app/static/tab_icons/([\w]+)\.svg", CSS))
        self.assertEqual(len(used), 15)
        self.assertEqual(used, {p.stem for p in ICONS.glob("*.svg")})
        for name in used:
            svg = (ICONS / f"{name}.svg").read_text()
            self.assertIn('viewBox="0 0 24 24"', svg)
            self.assertIn('fill="none" stroke="currentColor" stroke-width="1.6"', svg)
            self.assertIn('stroke-linecap="round" stroke-linejoin="round"', svg)
            self.assertNotRegex(svg, r"<(image|text|script)|#[0-9A-Fa-f]{3,6}")

    def test_tab_tile_slides_with_streamlits_selection_marker(self):
        # The marker copies the old tab's place and width, then moves by this transition.
        self.assertRegex(
            CSS,
            r"\.react-aria-SelectionIndicator \{[^}]*transition: translate var\(--cc-slide-time\) "
            r"ease-out, width var\(--cc-slide-time\) ease-out;",
        )


if __name__ == "__main__":
    unittest.main()
