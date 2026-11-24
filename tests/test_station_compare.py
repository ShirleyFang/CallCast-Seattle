"""Station comparison tests: the tables behind the Explore comparison charts, the average
station, the 3-station limit, and reading the clicked areas from the map.

Run: uv run python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd

from app_data import known_history, load_tables
from station_areas import AREA_LAYER_ID, selected_stations
from station_compare import (
    AVERAGE,
    BAR_COLOR,
    CITY,
    CITY_COLOR,
    COLORS,
    MAX_COMPARE,
    average_station,
    category_chart,
    category_means,
    category_shares,
    citywide,
    monthly,
    monthly_means,
    station_bar_chart,
    station_color,
    station_means,
    summary,
    trend,
    trend_chart,
    weekday_chart,
    weekday_means,
)

PICKED = ["SFD 10", "SFD 2", "SFD 41"]


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.counts, _ = load_tables()
        # Planning Oct 24: finished shifts through Oct 21, no missing-data shifts.
        cls.history = known_history(cls.counts, "2026-10-24")

    def test_up_to_three_stations(self):
        self.assertEqual(MAX_COMPARE, 3)
        self.assertEqual(len(COLORS), MAX_COMPARE)

    def test_average_station_is_citywide_over_34(self):
        avg = average_station(self.history).set_index("ShiftDate")
        city = self.history.groupby("ShiftDate").calls.sum()
        np.testing.assert_allclose(avg.calls, city / 34)
        self.assertTrue((avg.Station == AVERAGE).all())

    def test_summary_matches_the_counts(self):
        table = summary(self.history, PICKED)
        self.assertEqual(list(table.index), [*PICKED, AVERAGE])
        sfd10 = self.history[self.history.Station == "SFD 10"]
        week = sfd10[sfd10.ShiftDate >= "2026-10-15"].calls.mean()
        self.assertAlmostEqual(table.loc["SFD 10", "Last week"], round(week, 1))
        self.assertAlmostEqual(table.loc["SFD 10", "All history"], round(sfd10.calls.mean(), 1))
        self.assertIn(table.loc["SFD 10", "Busiest weekday"], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    def test_category_shares_add_up_to_100(self):
        shares = category_shares(self.history, PICKED)
        self.assertEqual(list(shares.index), [*PICKED, AVERAGE])
        np.testing.assert_allclose(shares.sum(axis=1), 100)
        self.assertEqual(list(shares.columns), ["Fire", "Medical", "Non-emergency", "Other"])

    def test_weekday_means_run_monday_to_sunday(self):
        table = weekday_means(self.history, PICKED)
        self.assertEqual(list(table.index), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        sfd2 = self.history[self.history.Station == "SFD 2"]
        monday = sfd2[sfd2.ShiftDate.dt.dayofweek == 0].calls.mean()
        self.assertAlmostEqual(table.loc["Mon", "SFD 2"], monday)

    def test_trend_covers_the_last_90_known_shifts(self):
        table = trend(self.history, PICKED)
        self.assertEqual(len(table), 90)
        self.assertEqual(table.index.max(), pd.Timestamp("2026-10-21"))
        self.assertEqual(list(table.columns), [*PICKED, AVERAGE])
        # Later counts cannot change it.
        altered = self.counts.copy()
        altered.loc[altered.ShiftDate >= "2026-10-22", "calls"] = 999999
        pd.testing.assert_frame_equal(table, trend(known_history(altered, "2026-10-24"), PICKED))

    def test_citywide_is_the_sum_of_all_stations(self):
        city = citywide(self.history).set_index("ShiftDate")
        np.testing.assert_array_equal(city.calls, self.history.groupby("ShiftDate").calls.sum())
        self.assertTrue((city.Station == CITY).all())
        # One line only: no average station next to the city total.
        table = summary(citywide(self.history), [CITY], average=False)
        self.assertEqual(list(table.index), [CITY])
        self.assertAlmostEqual(table.loc[CITY, "All history"], round(city.calls.mean(), 1))
        shares = category_shares(citywide(self.history), [CITY], average=False)
        np.testing.assert_allclose(shares.sum(axis=1), 100)
        self.assertEqual(list(trend(citywide(self.history), [CITY], average=False).columns), [CITY])

    def test_monthly_is_the_mean_per_shift(self):
        city = citywide(self.history)
        months = monthly(city)
        oct26 = city[city.ShiftDate.dt.to_period("M") == "2026-10"].calls.mean()
        self.assertAlmostEqual(months[pd.Timestamp("2026-10-01")], oct26)

    def test_station_bar_chart_highlights_the_picked_stations(self):
        ids = sorted(self.history.Station.unique(), key=lambda s: int(s.split()[-1]))
        means = station_means(self.history, ids)
        self.assertEqual(list(means.index), ids)
        bars = station_bar_chart(means, ["SFD 10", "SFD 2"]).data[0]
        colors = dict(zip(bars.x, bars.marker.color))
        self.assertEqual(colors["SFD 10"], COLORS[0])
        self.assertEqual(colors["SFD 2"], COLORS[1])
        self.assertEqual(colors["SFD 41"], BAR_COLOR)
        self.assertEqual(set(station_bar_chart(means).data[0].marker.color), {BAR_COLOR})

    def test_citywide_charts_use_the_neutral_colour(self):
        # Colour means a picked station; the whole city is drawn in slate.
        city = citywide(self.history)
        line = trend_chart(trend(city, [CITY], average=False)).data[0]
        bars = weekday_chart(weekday_means(city, [CITY], average=False)).data[0]
        self.assertEqual(line.line.color, CITY_COLOR)
        self.assertEqual(bars.marker.color, CITY_COLOR)
        self.assertNotIn(CITY_COLOR, COLORS)
        self.assertNotIn(BAR_COLOR, COLORS)

    def test_monthly_means_per_station(self):
        # The long-term view the sidebar-station chart used to show, now per picked station.
        table = monthly_means(self.history, PICKED)
        self.assertEqual(list(table.columns), [*PICKED, AVERAGE])
        self.assertEqual(table.index.min(), pd.Timestamp("2022-01-01"))
        sfd10 = self.history[self.history.Station == "SFD 10"]
        old = sfd10.groupby(sfd10.ShiftDate.dt.to_period("M")).calls.mean()
        old.index = old.index.to_timestamp()
        pd.testing.assert_series_equal(table["SFD 10"].dropna(), old, check_names=False, check_freq=False)

    def test_category_counts_match_the_old_category_chart(self):
        # The old Explore chart showed part[GROUPS].mean(): average calls per shift per category.
        means = category_means(self.history, PICKED)
        sfd10 = self.history[self.history.Station == "SFD 10"]
        old = sfd10[["fire", "medical", "non_emergency", "other"]].mean()
        np.testing.assert_allclose(means.loc["SFD 10"].to_numpy(), old.to_numpy())
        # The hover text shows both the share and that number.
        bars = category_chart(category_shares(self.history, PICKED), means).data
        medical = next(b for b in bars if b.name == "Medical")
        self.assertIn("calls per shift", medical.hovertemplate)
        np.testing.assert_allclose(medical.customdata, means["Medical"].to_numpy())

    def test_map_colours_match_the_chart_colours(self):
        self.assertEqual(station_color(PICKED, "SFD 2"), COLORS[1])
        self.assertEqual(station_color(PICKED, "SFD 10", alpha=150), [47, 93, 138, 150])


class CitywideScreenTests(unittest.TestCase):
    def test_citywide_tab_ignores_the_sidebar_station(self):
        # Explore starts with the whole city; the sidebar station only drives the Prediction tab.
        import os

        os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
        from streamlit.testing.v1 import AppTest

        from app_data import ROOT

        def citywide_table(app):
            city = next(t for t in app.tabs if t.label == "Citywide")
            return city.dataframe[0].value

        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        before = citywide_table(app)
        self.assertEqual(list(before.index), [CITY])
        app.sidebar.selectbox[0].select("SFD 2").run(timeout=60)
        pd.testing.assert_frame_equal(before, citywide_table(app))
        # With nothing picked on the map, the Stations tab points to the map.
        stations = next(t for t in app.tabs if t.label == "Stations")
        self.assertTrue(any("Click up to 3 station areas" in i.value for i in stations.info))


class SelectionTests(unittest.TestCase):
    def test_clicked_areas_are_read_in_order_without_repeats(self):
        def area(name):
            return {"properties": {"Station": name}}

        event = {"selection": {"objects": {AREA_LAYER_ID: [area("SFD 5"), area("SFD 22"), area("SFD 5")],
                                           "station-dots": [{"station": "SFD 9"}]}}}
        self.assertEqual(selected_stations(event), ["SFD 5", "SFD 22"])
        self.assertEqual(selected_stations(None), [])
        self.assertEqual(selected_stations({"selection": {"objects": {}}}), [])


if __name__ == "__main__":
    unittest.main()
