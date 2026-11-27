"""Explore > Weather data tests: the saved weather effects (scripts/make_weather_effects.py)
are complete, add up, use 2022-2025 only, and the past events show in ERA5.

Run: uv run python -m unittest discover -s tests -v
"""

import unittest

import pandas as pd

from app_data import ROOT

DATA = ROOT / "data"


class WeatherEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.effects = pd.read_csv(DATA / "weather_effects.csv")
        cls.events = pd.read_csv(DATA / "weather_events.csv", parse_dates=["start", "end"])
        cls.shifts = pd.read_csv(DATA / "weather_event_shifts.csv", parse_dates=["ShiftDate"])

    def test_groups_in_order(self):
        temperature = self.effects[self.effects["chart"] == "temperature"].sort_values("order")
        self.assertEqual(temperature["group"].tolist(), [
            "Low -2 °C or colder", "Low -2 to 0 °C", "Low 0 to 2 °C",
            "High 30 °C or hotter", "High 25 to 30 °C", "Mild"])
        spell = self.effects[self.effects["chart"] == "cold spell"].sort_values("order")
        self.assertEqual(spell["group"].tolist(), [
            "Single cold day", "Day 1 of a cold spell", "Day 2", "Day 3", "Day 4 or later"])

    def test_single_cold_day_hover_names_no_spell(self):
        # The single cold day bar reads "5 days", not "5 days in 5 cold spells"; Day 4 or
        # later keeps "14 days in 6 cold spells".
        from weather_explore import spell_chart

        hover = list(spell_chart(self.effects).data[0].customdata)
        self.assertTrue(hover[0].startswith("5 days<br>"), hover[0])
        self.assertTrue(hover[-1].startswith("14 days in 6 cold spells"), hover[-1])

    def test_temperature_groups_cover_every_non_holiday_shift_once(self):
        # 1,450 shifts in 2022-2025 with all stations, 96 of them holidays or special days
        temperature = self.effects[self.effects["chart"] == "temperature"]
        self.assertEqual(temperature["shifts"].sum(), 1450 - 96)

    def test_extra_calls_are_calls_minus_usual(self):
        diff = self.effects["calls"] - self.effects["usual_calls"] - self.effects["extra_calls"]
        self.assertLessEqual(diff.abs().max(), 0.11)

    def test_only_2022_to_2025(self):
        self.assertTrue((self.effects["years"] == "2022-2025").all())
        self.assertLess(self.shifts["ShiftDate"].max(), pd.Timestamp("2026-01-01"))
        self.assertGreaterEqual(self.shifts["ShiftDate"].min(), pd.Timestamp("2022-01-01"))

    def test_events_match_their_shifts(self):
        for _, e in self.events.iterrows():
            with self.subTest(event=e["event"]):
                rows = self.shifts[self.shifts["event"] == e["event"]]
                inside = rows[rows["in_event"]]
                self.assertEqual(len(inside), e["shifts"])
                self.assertEqual(inside["ShiftDate"].min(), e["start"])
                self.assertEqual(inside["ShiftDate"].max(), e["end"])
                self.assertAlmostEqual(inside["calls"].mean(), e["calls"], delta=0.06)
                # a week on each side
                self.assertEqual(rows["ShiftDate"].min(), e["start"] - pd.Timedelta(days=7))
                self.assertEqual(rows["ShiftDate"].max(), e["end"] + pd.Timedelta(days=7))

    def test_events_show_in_era5(self):
        cold = self.events[self.events["kind"] == "cold"]
        self.assertTrue((cold["lowest_low"] <= -5).all())
        heat = self.events[self.events["kind"] == "heat"]
        self.assertTrue((heat["highest_high"] >= 30).all())
        wind = self.events[self.events["kind"] == "wind"]
        self.assertTrue((wind["strongest_gust"] >= 70).all())

    def test_notes_are_plain(self):
        for note in self.events["note"]:
            for word in ["ERA5", "model", "MAE", "%"]:
                self.assertNotIn(word, note)


if __name__ == "__main__":
    unittest.main()
