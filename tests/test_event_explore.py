"""Explore > Events tests: the saved event effects match the event analysis, the per-event
rows add up to them, and the Events sub-tab shows both tables.

Run: uv run python -m unittest discover -s tests -v
"""

import os
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
os.environ.setdefault("CALLCAST_OFFLINE", "1")  # also in tests/__init__.py

from streamlit.testing.v1 import AppTest

from app_data import ROOT
from text_blocks import plain
from chart_theme import event_impact_chart
from event_explore import affected_features, effects_table, load_event_effects, venue_map, venue_rows
from station_areas import build_station_areas, load_boundary

# The three pairs chosen on 2022-2024 (docs/POST_MODEL_ADJUSTMENT.md), with their extra calls
# per event day measured on 2022-2025 (the years the final model is trained on).
CLEAR = {
    ("Husky football", "Husky Stadium", "SFD 22"): 3.95,
    ("Seahawks", "Lumen Field", "SFD 10"): 4.58,
    ("stadium concert", "Lumen Field", "SFD 10"): 4.37,
}


class EventEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.effects = load_event_effects()
        cls.shifts = pd.read_csv(ROOT / "data/event_shifts.csv", parse_dates=["ShiftDate"])

    def test_three_clear_pairs_with_2022_2025_numbers(self):
        clear = self.effects[self.effects.clear].set_index(["kind", "venue", "station"])
        self.assertEqual(set(clear.index), set(CLEAR))
        for key, extra in CLEAR.items():
            self.assertAlmostEqual(clear.loc[key, "extra_calls"], extra, places=2)
        self.assertTrue((self.effects.years == "2022-2025").all())
        self.assertTrue((self.effects.chosen_on == "2022-2024").all())
        # With more years the three pairs still pass the Step 14 rule.
        self.assertTrue(((clear.events >= 10) & (clear.extra_pct >= 10) & (clear.t >= 2)).all())

    def test_pairs_are_still_chosen_on_2022_2024(self):
        # "Clear" is the selection from the event analysis, not re-chosen on newer years.
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from cv_utils import load_dataset
        from feature_sets import CHOOSE_END, choose_event_pairs, load_events

        df = load_dataset(str(ROOT / "data/modeling_dataset.csv"))  # feature_sets reads data/ from the repo root
        _, chosen = choose_event_pairs(df[df.ShiftDate < CHOOSE_END], load_events())
        self.assertEqual(set(chosen), set(CLEAR))

    def test_extra_calls_by_type_add_up(self):
        # The four types add up to all extra calls; at the clear pairs the extra calls are mostly
        # medical, and fire calls barely change.
        types = ["extra_fire", "extra_medical", "extra_non_emergency", "extra_other"]
        np.testing.assert_allclose(self.effects[types].sum(axis=1), self.effects.extra_calls, atol=0.02)
        clear = self.effects[self.effects.clear]
        self.assertTrue((clear.extra_medical > clear[["extra_fire", "extra_other"]].max(axis=1)).all())
        self.assertTrue((clear.extra_fire.abs() < 0.5).all())

    def test_per_event_rows_add_up_to_the_summary(self):
        # 2022-2025 only: the 2026 test year is never used.
        self.assertTrue((self.shifts.ShiftDate < "2026-01-01").all())
        self.assertGreaterEqual(self.shifts.ShiftDate.max(), pd.Timestamp("2025-01-01"))
        by_pair = self.shifts.groupby(["kind", "venue", "station"]).difference.agg(["size", "mean"])
        summary = self.effects.set_index(["kind", "venue", "station"]).join(by_pair)
        np.testing.assert_array_equal(summary["size"], summary["events"])
        np.testing.assert_allclose(summary["mean"], summary["extra_calls"], atol=0.01)
        np.testing.assert_allclose(self.shifts.calls - self.shifts.usual_calls, self.shifts.difference, atol=0.01)

    def test_impact_chart_has_one_bar_per_row(self):
        # Stadium concerts at Lumen Field and T-Mobile Park both sit next to SFD 5; each
        # keeps its own bar, so SFD 5's Lumen bar is +1.55, not the two added (+2.48).
        bars = event_impact_chart(self.effects).data[0]
        self.assertEqual(len(set(bars.y)), len(self.effects))
        lumen = list(bars.y).index("Stadium concert at Lumen Field (SFD 5)")
        self.assertAlmostEqual(bars.x[lumen], 1.55)

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
        self.assertIn("Seahawks → SFD 10: +4.6 calls per event day (35 events), mostly medical (+3.5) "
                      "and non-emergency (+1.1)", venues.loc["Lumen Field", "body"])
        self.assertIn("mostly medical (+2.2), non-emergency (+1.0) and other (+0.6)",
                      venues.loc["Husky Stadium", "body"])
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
        self.assertEqual(labels["fontFamily"], "Figtree, Helvetica Neue, Arial, sans-serif")


class EventScreenTests(unittest.TestCase):
    def test_events_sub_tab_shows_both_tables(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        events = next(t for t in app.tabs if t.label == "Events")
        self.assertEqual(len(events.get("deck_gl_json_chart")), 1)  # the venue map
        self.assertEqual(len(events.dataframe), 2)
        self.assertEqual(len(events.dataframe[0].value), 3)  # the clear pairs
        blocks = [plain(h.value) for h in events.get("html")]
        text = " ".join(blocks)
        self.assertIn("Based on 2022-2025", text)  # the numbers
        self.assertIn("chosen on 2022-2024", text)  # the pair selection
        self.assertNotIn("2026", text)
        self.assertIn("These are averages", text)
        # Unlike Citywide and Stations, the event numbers are fixed and say so.
        self.assertIn("Past event days from Jan 2022 to Dec 2025", blocks[0])
        self.assertIn("do not change with the selected date", blocks[0])
        before = events.dataframe[0].value
        from datetime import date

        app.date_input[0].set_value(date(2026, 3, 10)).run(timeout=60)
        later = next(t for t in app.tabs if t.label == "Events").dataframe[0].value
        pd.testing.assert_frame_equal(before, later)


if __name__ == "__main__":
    unittest.main()
