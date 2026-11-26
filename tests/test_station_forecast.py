"""Prediction tab pieces: the watch list, the map shades, the call type mix, picking stations
on the map and the freshness line.

Run: uv run python -m unittest discover -s tests -v
"""

import unittest

import pandas as pd

from station_forecast import (
    WATCH_MAX,
    data_through,
    difference_fills,
    mix_table,
    season_mix,
    stations_to_watch,
    toggle_picks,
)


def forecast(rows):
    # (station, difference, usual) -> the columns the watch list and the shading read.
    out = pd.DataFrame(rows, columns=["Station", "difference", "usual_calls"])
    return out.assign(predicted_calls=out["usual_calls"] + out["difference"])


class WatchListTests(unittest.TestCase):
    def test_needs_both_the_calls_and_the_share(self):
        view = forecast(
            [
                ("above", 6, 30),  # 6 calls and 20%: in
                ("small", 1.9, 5),  # 38% but under 2 calls: out
                ("share", 3, 40),  # 3 calls but 7.5%: out
                ("below", -5, 30),
            ]
        )
        self.assertEqual(list(stations_to_watch(view)["Station"]), ["above"])

    def test_exact_threshold_counts(self):
        # Exactly 2 calls and exactly 10% of the usual is in; a hair under either is out.
        view = forecast([("edge", 2.0, 20), ("calls", 1.99, 10), ("share", 2.0, 20.1)])
        self.assertEqual(list(stations_to_watch(view)["Station"]), ["edge"])

    def test_none_above_usual_gives_an_empty_list(self):
        view = forecast([("a", 0.5, 10), ("b", -3, 20)])
        self.assertTrue(stations_to_watch(view).empty)

    def test_shows_the_biggest_gaps_first_and_at_most_five(self):
        view = forecast([(f"s{i}", 3 + i, 10) for i in range(8)])
        shown = list(stations_to_watch(view)["Station"])
        self.assertEqual(len(shown), WATCH_MAX)
        self.assertEqual(shown, ["s7", "s6", "s5", "s4", "s3"])


class ShadeTests(unittest.TestCase):
    def test_blue_below_light_at_usual_orange_above(self):
        fills = difference_fills(forecast([("q", -4, 20), ("u", 0, 20), ("b", 4, 20)]))
        self.assertEqual(fills["q"], [91, 141, 184, 210])
        self.assertEqual(fills["u"], [244, 241, 234, 210])
        self.assertEqual(fills["b"], [200, 100, 26, 210])

    def test_both_sides_use_the_same_span(self):
        fills = difference_fills(forecast([("q", -2, 20), ("b", 4, 20)]))
        # +4 sets the span, so -2 is half way down the blue side (the second stop), not the end.
        self.assertEqual(fills["b"], [200, 100, 26, 210])
        self.assertEqual(fills["q"][:3], [185, 211, 232])

    def test_tiny_gaps_do_not_fill_the_whole_scale(self):
        # With every gap under 2 calls, +1 reaches only half way up the orange side.
        fills = difference_fills(forecast([("a", 1, 20), ("b", -1, 20)]))
        self.assertEqual(fills["a"][:3], [244, 195, 141])
        self.assertNotEqual(fills["a"][:3], [200, 100, 26])


class CallTypeMixTests(unittest.TestCase):
    def setUp(self):
        # Station A has two Octobers in earlier years; B only one. Nov 2025 and Oct 2026
        # are not the target's season or not an earlier year.
        rows = [
            ("A", "2023-10-01", 4, 1, 0, 1),
            ("B", "2023-10-01", 2, 0, 0, 0),
            ("A", "2023-10-02", 6, 1, 0, 1),
            ("B", "2023-10-02", 2, 2, 0, 0),
            ("A", "2025-10-05", 8, 4, 1, 1),
            ("B", "2025-10-05", 2, 0, 1, 0),
            ("A", "2025-11-05", 90, 90, 90, 90),
            ("B", "2025-11-05", 90, 90, 90, 90),
            ("A", "2026-10-05", 90, 90, 90, 90),
            ("B", "2026-10-05", 90, 90, 90, 90),
        ]
        cols = ["Station", "ShiftDate", "medical", "fire", "non_emergency", "other"]
        self.history = pd.DataFrame(rows, columns=cols).assign(
            ShiftDate=lambda d: pd.to_datetime(d["ShiftDate"])
        )

    def test_station_averages_are_for_the_same_months_of_earlier_years(self):
        mix = season_mix(self.history, "2026-10-24")
        # A medical: (4 + 6 + 8) / 3 = 6; fire: (1 + 1 + 4) / 3 = 2.
        self.assertEqual(mix.loc["A", "Medical"], 6)
        self.assertEqual(mix.loc["A", "Fire"], 2)
        self.assertEqual(mix.loc["B", "Medical"], 2)

    def test_citywide_row_adds_the_stations_per_shift(self):
        mix = season_mix(self.history, "2026-10-24")
        # Medical per shift: 6, 8 and 10 across the three shifts: average 8.
        self.assertEqual(mix.loc["Citywide", "Medical"], 8)
        # Fire: 1, 3 and 4 -> 8/3.
        self.assertAlmostEqual(mix.loc["Citywide", "Fire"], 8 / 3)

    def test_table_has_a_row_per_type_and_a_column_per_name(self):
        table = mix_table(season_mix(self.history, "2026-10-24"), ["B", "A"])
        self.assertEqual(list(table.columns), ["B", "A"])
        self.assertEqual(
            list(table.index), ["Medical", "Fire", "Non-emergency", "Other"]
        )
        self.assertEqual(table.loc["Fire", "A"], 2.0)


class PickTests(unittest.TestCase):
    def test_a_click_picks_and_a_second_click_unpicks(self):
        self.assertEqual(toggle_picks([], ["SFD 5"]), (["SFD 5"], []))
        self.assertEqual(toggle_picks(["SFD 5", "SFD 2"], ["SFD 5"]), (["SFD 2"], []))

    def test_picking_keeps_click_order(self):
        picked, _ = toggle_picks(["SFD 10"], ["SFD 2"])
        self.assertEqual(picked, ["SFD 10", "SFD 2"])

    def test_a_fourth_station_is_refused(self):
        picked, refused = toggle_picks(["a", "b", "c"], ["d"])
        self.assertEqual((picked, refused), (["a", "b", "c"], ["d"]))

    def test_a_pick_makes_room_for_the_next_click_in_one_batch(self):
        picked, refused = toggle_picks(["a", "b", "c"], ["a", "d"])
        self.assertEqual((picked, refused), (["b", "c", "d"], []))


class FreshnessTests(unittest.TestCase):
    def test_data_runs_to_the_end_of_the_last_finished_shift(self):
        # The Wednesday shift starts at 08:00 Wednesday and ends at 08:00 Thursday.
        history = pd.DataFrame(
            {"ShiftDate": pd.to_datetime(["2026-10-20", "2026-10-21"])}
        )
        self.assertEqual(data_through(history), "Thu Oct 22 08:00")

    def test_a_shift_with_missing_data_pulls_the_time_back(self):
        history = pd.DataFrame(
            {"ShiftDate": pd.to_datetime(["2026-10-19", "2026-10-20"])}
        )
        self.assertEqual(data_through(history), "Wed Oct 21 08:00")


if __name__ == "__main__":
    unittest.main()
