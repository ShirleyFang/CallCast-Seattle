"""Station area map tests: the areas fill the city without overlapping, follow the
nearest-station rule used to give calls to stations, and show recent averages on hover.

Run: uv run python -m unittest discover -s tests -v
"""

import json
import os
import unittest
from datetime import date

import numpy as np
import pandas as pd
from shapely import Point
from sklearn.neighbors import BallTree

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")

from streamlit.testing.v1 import AppTest

from app_data import ROOT, known_history, load_tables
from station_areas import (
    area_features,
    area_km2,
    build_station_areas,
    load_boundary,
    recent_averages,
)


class StationAreaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stations = pd.read_csv(ROOT / "data/sfd_stations.csv")
        cls.boundary = load_boundary()
        cls.areas = build_station_areas(cls.stations, cls.boundary)

    def test_one_area_per_station_filling_the_city(self):
        self.assertEqual(set(self.areas), set(self.stations.station))
        self.assertTrue(all(not a.is_empty and a.is_valid for a in self.areas.values()))
        total = sum(area_km2(a) for a in self.areas.values())
        self.assertAlmostEqual(total, area_km2(self.boundary), delta=0.01)
        # Each station sits inside its own area.
        for _, row in self.stations.iterrows():
            self.assertTrue(self.areas[row.station].contains(Point(row.lon, row.lat)), row.station)

    def test_city_boundary_is_cleaned_of_the_magnolia_spike(self):
        # The source ring has a 200 m spike folding back on itself on the Magnolia shore
        # (a 44 m2 sliver) that drew a stray line near SFD 41. The cleaned boundary is a
        # valid single polygon with no holes, the same area, and no spike.
        from shapely.geometry import shape

        raw = shape(json.loads((ROOT / "data/seattle_area.geojson").read_text())["features"][0]["geometry"])
        self.assertFalse(raw.is_valid)
        self.assertTrue(self.boundary.is_valid)
        self.assertEqual(self.boundary.geom_type, "Polygon")
        self.assertEqual(len(self.boundary.interiors), 0)
        self.assertAlmostEqual(area_km2(self.boundary), area_km2(raw), delta=0.001)
        spike_tip = Point(raw.exterior.coords[2])
        self.assertGreater(self.boundary.exterior.distance(spike_tip), 0.001)  # ~200 m in degrees

    def test_areas_do_not_overlap(self):
        names = list(self.areas)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = self.areas[a].intersection(self.areas[b])
                self.assertLess(area_km2(overlap), 1e-6, f"{a} and {b}")

    def test_areas_follow_the_nearest_station_rule(self):
        # The notebook gives each call to its nearest station (haversine BallTree). Random
        # points in the city should land in that station's area, except within a few metres
        # of a border, where the flat-map distance can pick the neighbour.
        rng = np.random.default_rng(0)
        minx, miny, maxx, maxy = self.boundary.bounds
        pts = np.column_stack([rng.uniform(minx, maxx, 4000), rng.uniform(miny, maxy, 4000)])
        pts = pts[[self.boundary.contains(Point(p)) for p in pts]]
        tree = BallTree(np.radians(self.stations[["lat", "lon"]].to_numpy()), metric="haversine")
        dist, idx = tree.query(np.radians(pts[:, ::-1]), k=2)
        nearest = self.stations.station.to_numpy()[idx[:, 0]]
        mine = np.array([next(k for k, a in self.areas.items() if a.covers(Point(p))) for p in pts])
        self.assertGreater(np.mean(mine == nearest), 0.99)
        gap_m = (dist[:, 1] - dist[:, 0]) * 6371000
        self.assertTrue((gap_m[mine != nearest] < 20).all())

    def test_recent_averages_use_only_known_shifts(self):
        # Planning Oct 24: the last finished shift is Oct 21, so "last week" is Oct 15-21.
        _, counts, _ = load_tables()
        history = known_history(counts, "2026-10-24")
        averages = recent_averages(history)
        self.assertEqual(averages["week"]["to"], pd.Timestamp("2026-10-21"))
        self.assertEqual(averages["week"]["from"], pd.Timestamp("2026-10-15"))
        self.assertEqual(averages["month"]["from"], pd.Timestamp("2026-09-24"))
        sfd10 = counts[(counts.Station == "SFD 10") & ~counts.Masked]
        week = sfd10[sfd10.ShiftDate.between("2026-10-15", "2026-10-21")].calls.mean()
        self.assertAlmostEqual(averages["week"]["mean"]["SFD 10"], week)
        # Later counts must not change the averages.
        altered = counts.copy()
        altered.loc[altered.ShiftDate >= "2026-10-22", "calls"] = 999999
        again = recent_averages(known_history(altered, "2026-10-24"))
        pd.testing.assert_series_equal(again["month"]["mean"], averages["month"]["mean"])
        # The hover text is ready to show.
        props = {f["properties"]["Station"]: f["properties"] for f in area_features(self.areas, averages)}
        self.assertEqual(props["SFD 10"]["week_dates"], "Oct 15 to Oct 21")
        self.assertEqual(props["SFD 10"]["week_avg"], f"{week:.1f}")


class StationAreaScreenTests(unittest.TestCase):
    def area_map(self, app):
        for chart in app.get("deck_gl_json_chart"):
            spec = json.loads(chart.proto.json)
            if any(layer["@@type"] == "GeoJsonLayer" for layer in spec["layers"]):
                return spec
        return None

    def test_explore_tab_shows_the_area_map(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception, [e.message for e in app.exception])
        spec = self.area_map(app)
        self.assertIsNotNone(spec, "no station area map on the page")
        layers = {layer["@@type"]: layer for layer in spec["layers"]}
        areas = next(l for l in spec["layers"] if l.get("pickable"))
        self.assertEqual(len(areas["data"]["features"]), 34)
        self.assertEqual(layers["ScatterplotLayer"]["data"].__len__(), 34)
        # Outlines only: the city boundary has no fill, and the station areas a nearly clear one.
        city = next(l for l in spec["layers"] if l["@@type"] == "GeoJsonLayer" and not l.get("pickable"))
        self.assertFalse(city["filled"])
        caption = " ".join(c.value for c in app.tabs[1].caption)
        self.assertIn("closest to that station", caption)

    def test_line_and_dot_sizes_reach_the_browser_as_numbers(self):
        # pydeck turns plain strings into JavaScript expressions, so a units setting such as
        # "pixels" arrived as an unknown name and drew huge dots and lines. Sizes are now
        # capped in pixels with numbers, and the only expression left is the dot position.
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        for layer in self.area_map(app)["layers"]:
            expressions = {
                k: v for k, v in layer.items() if isinstance(v, str) and v.startswith("@@=")
            }
            # Expected expressions: the dot position, and each area's own fill colour.
            allowed = {
                "station-dots": {"getPosition": "@@=[lon, lat]"},
                "station-areas": {"getFillColor": "@@=properties.fill"},
            }.get(layer["id"], {})
            self.assertEqual(expressions, allowed, layer["@@type"])
            if layer["@@type"] == "GeoJsonLayer":
                self.assertLessEqual(layer["lineWidthMaxPixels"], 3)
            else:
                self.assertLessEqual(layer["radiusMaxPixels"], 6)

    def test_future_dates_use_the_last_known_week(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        app.sidebar.date_input[0].set_value(date(2026, 12, 25)).run(timeout=60)
        self.assertFalse(app.exception)
        areas = next(l for l in self.area_map(app)["layers"] if l.get("pickable"))
        sfd10 = next(f["properties"] for f in areas["data"]["features"] if f["properties"]["Station"] == "SFD 10")
        self.assertEqual(sfd10["week_dates"], "Oct 22 to Oct 28")


if __name__ == "__main__":
    unittest.main()
