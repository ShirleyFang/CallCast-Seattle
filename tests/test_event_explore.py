"""Explore > Events tests: the saved event effects match the event analysis, the per-event
rows add up to them, and the Events sub-tab shows both tables.

Run: uv run python -m unittest discover -s tests -v
"""

import os
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")

from streamlit.testing.v1 import AppTest

from app_data import ROOT
from event_explore import affected_features, effects_table, load_event_effects, venue_map, venue_rows
from station_areas import build_station_areas, load_boundary

CLEAR = {
    ("Husky football", "Husky Stadium", "SFD 22"): 4.40,
    ("Seahawks", "Lumen Field", "SFD 10"): 3.84,
    ("stadium concert", "Lumen Field", "SFD 10"): 4.13,
}


class EventEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.effects = load_event_effects()
        cls.shifts = pd.read_csv(ROOT / "data/event_shifts.csv", parse_dates=["ShiftDate"])

    def test_three_clear_pairs_match_the_event_analysis(self):
        # The same pairs and extra calls as docs/POST_MODEL_ADJUSTMENT.md (2022-2024).
        clear = self.effects[self.effects.clear].set_index(["kind", "venue", "station"])
        self.assertEqual(set(clear.index), set(CLEAR))
        for key, extra in CLEAR.items():
            self.assertAlmostEqual(clear.loc[key, "extra_calls"], extra, places=2)
        self.assertTrue((self.effects.years == "2022-2024").all())

    def test_clear_means_the_step_14_rule(self):
        rule = (self.effects.events >= 10) & (self.effects.extra_pct >= 10) & (self.effects.t >= 2)
        pd.testing.assert_series_equal(self.effects.clear, rule, check_names=False)

    def test_per_event_rows_add_up_to_the_summary(self):
        self.assertTrue((self.shifts.ShiftDate < "2025-01-01").all())
        by_pair = self.shifts.groupby(["kind", "venue", "station"]).difference.agg(["size", "mean"])
        summary = self.effects.set_index(["kind", "venue", "station"]).join(by_pair)
        np.testing.assert_array_equal(summary["size"], summary["events"])
        np.testing.assert_allclose(summary["mean"], summary["extra_calls"], atol=0.01)
        np.testing.assert_allclose(self.shifts.calls - self.shifts.usual_calls, self.shifts.difference, atol=0.01)

    def test_table_uses_plain_names(self):
        table = effects_table(self.effects)
        self.assertEqual(
            list(table.columns),
            ["Event", "Venue", "Station", "Events", "Usual calls", "Calls on event days", "Extra calls", "Extra %"],
        )
        self.assertIn("Stadium concert", set(table.Event))  # not the raw "stadium concert"
        self.assertTrue(table["Extra %"].is_monotonic_decreasing)


class VenueMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.effects = load_event_effects()
        cls.boundary = load_boundary()
        cls.areas = build_station_areas(pd.read_csv(ROOT / "data/sfd_stations.csv"), cls.boundary)

    def test_four_venues_with_their_events(self):
        venues = venue_rows(self.effects).set_index("venue")
        self.assertEqual(set(venues.index), {"Lumen Field", "T-Mobile Park", "Climate Pledge Arena", "Husky Stadium"})
        self.assertIn("Seahawks → SFD 10: +3.8 calls per event day (27 events)", venues.loc["Lumen Field", "body"])
        self.assertIn("No clear effect: Sounders", venues.loc["Lumen Field", "body"])
        self.assertTrue(venues.loc["T-Mobile Park", "body"].startswith("No clear effect"))
        # Lumen Field and T-Mobile Park are 400 m apart: labels go on opposite sides.
        self.assertNotEqual(venues.loc["Lumen Field", "offset"], venues.loc["T-Mobile Park", "offset"])
        # Plain-text hover: lines are split with newlines, never with HTML tags.
        self.assertFalse(venues["body"].str.contains("<").any())

    def test_only_clear_stations_are_coloured(self):
        features = {f["properties"]["title"]: f["properties"] for f in affected_features(self.effects, self.areas)}
        self.assertEqual(set(features), {"SFD 10", "SFD 22"})
        # The two areas get different colours.
        self.assertNotEqual(features["SFD 10"]["fill"], features["SFD 22"]["fill"])

    def test_map_settings_reach_the_browser_as_intended(self):
        # Strings become JavaScript expressions in pydeck unless wrapped in matching quotes;
        # only the field lookups below are meant as expressions.
        import json

        spec = json.loads(venue_map(self.effects, self.areas, self.boundary).to_json())
        allowed = {"getPosition", "getText", "getPixelOffset", "getFillColor", "getLineColor"}
        for layer in spec["layers"]:
            for key, value in layer.items():
                if isinstance(value, str) and value.startswith("@@="):
                    self.assertIn(key, allowed, f"{layer['id']}.{key} = {value}")
        labels = next(l for l in spec["layers"] if l["id"] == "event-venue-labels")
        self.assertEqual(labels["fontFamily"], "Public Sans, Helvetica Neue, Arial, sans-serif")


class EventScreenTests(unittest.TestCase):
    def test_events_sub_tab_shows_both_tables(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        events = next(t for t in app.tabs if t.label == "Events")
        self.assertEqual(len(events.get("deck_gl_json_chart")), 1)  # the venue map
        self.assertEqual(len(events.dataframe), 2)
        self.assertEqual(len(events.dataframe[0].value), 3)  # the clear pairs
        text = " ".join(c.value for c in events.caption)
        self.assertIn("2022-2024", text)
        self.assertIn("These are averages", text)


if __name__ == "__main__":
    unittest.main()
