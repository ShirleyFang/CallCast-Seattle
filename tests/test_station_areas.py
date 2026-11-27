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
from shapely import Point, box, transform
from sklearn.neighbors import BallTree

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
os.environ.setdefault("CALLCAST_OFFLINE", "1")  # also in tests/__init__.py

from streamlit.testing.v1 import AppTest

from app_data import ROOT, known_history, load_tables
from chart_theme import DAY_STOPS, rgb
from text_blocks import plain
from station_areas import (
    AREA_LAYER_ID,
    CITY_PAD_Y,
    MAP_MIN_WIDTH,
    MAP_PAD,
    NEARBY_PLACES,
    area_features,
    area_geometries,
    area_km2,
    area_labels,
    build_station_areas,
    fixed_view,
    label_size,
    forecast_map,
    load_boundary,
    mercator,
    picked_outline_layers,
    recent_averages,
    shade,
    value_map,
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

    def test_shading_runs_from_mint_to_rose(self):
        # Explore maps use the day scale: the fewest calls mint, the most rose.
        fills = shade({"low": 1.0, "mid": 5.0, "high": 9.0})
        self.assertEqual(fills["low"][:3], rgb(DAY_STOPS[0]))
        self.assertEqual(fills["high"][:3], rgb(DAY_STOPS[-1]))
        # The citywide map colours every area (clicking one opens its history window in app.py);
        # picked stations elsewhere get a halo and a coloured outline.
        values = pd.Series({name: float(i) for i, name in enumerate(self.areas)})
        spec = json.loads(value_map(self.areas, self.boundary, values, {}).to_json())
        areas_layer = next(l for l in spec["layers"] if l["id"] == AREA_LAYER_ID)
        self.assertEqual(len(areas_layer["data"]["features"]), 34)
        outlines = picked_outline_layers(self.areas, {"SFD 10": [47, 93, 138, 255]})
        self.assertEqual([layer.id for layer in outlines], ["picked-halo", "picked-outline"])

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

    def test_fixed_view_fits_city_and_town_names_into_the_phone_frame(self):
        view, height = fixed_view(self.boundary)
        scale = 2 ** view["zoom"]
        cx, cy = mercator(view["longitude"], view["latitude"])

        def to_px(lon, lat):
            x, y = mercator(lon, lat)
            return (x - cx) * scale + MAP_MIN_WIDTH / 2, (y - cy) * scale + height / 2

        # The whole city is inside, with at least CITY_PAD_Y px of land above and below it.
        w, s, e, n = self.boundary.bounds
        (left, top), (right, bottom) = to_px(w, n), to_px(e, s)
        self.assertGreaterEqual(left, MAP_PAD - 0.01)
        self.assertLessEqual(right, MAP_MIN_WIDTH - MAP_PAD + 0.01)
        self.assertGreaterEqual(top, CITY_PAD_Y)
        self.assertLessEqual(bottom, height - CITY_PAD_Y)
        # Each town name sits on its town, inside the frame, clear of the station areas, the
        # other names and the map's info button in the bottom-right corner.
        city = transform(self.boundary, lambda c: np.array([to_px(*p) for p in c]))
        info = box(MAP_MIN_WIDTH - 36, height - 36, MAP_MIN_WIDTH, height)
        names = [t for t, _, _ in NEARBY_PLACES]
        self.assertTrue({"Shoreline", "Bellevue", "Burien"} <= set(names))
        boxes = []
        for name, lon, lat in NEARBY_PLACES:
            x, y = to_px(lon, lat)
            wd, ht = label_size(name)
            label = box(x - wd / 2, y - ht / 2, x + wd / 2, y + ht / 2)
            self.assertTrue(box(0, 0, MAP_MIN_WIDTH, height).contains(label), name)
            self.assertGreater(label.distance(city), 3, name)
            self.assertGreater(label.distance(info), 3, name)
            self.assertTrue(all(label.distance(other) > 2 for other in boxes), name)
            boxes.append(label)

    def test_forecast_map_is_flat_fixed_on_pale_basemap_and_labels_areas(self):
        geoms, rings = area_geometries(self.areas)
        labels = area_labels(self.areas)
        fills = {name: [200, 100, 26, 210] for name in self.areas}
        expected = {name: 15.0 for name in self.areas}
        spec = json.loads(
            forecast_map(
                geoms, rings, self.boundary, fills, labels, expected=expected,
                picked_colors={"SFD 10": [47, 93, 138, 255]},
            ).to_json()
        )
        # Streamlit forces the controller on, so the view state itself pins the map: one zoom,
        # flat, and no drag, wheel, pinch, double-click or key moves.
        view = spec["initialViewState"]
        self.assertEqual(view["minZoom"], view["zoom"])
        self.assertEqual(view["maxZoom"], view["zoom"])
        self.assertEqual([view["pitch"], view["minPitch"], view["maxPitch"], view["bearing"]], [0] * 4)
        for flag in ["dragPan", "dragRotate", "scrollZoom", "doubleClickZoom", "touchZoom",
                     "touchRotate", "keyboard"]:
            self.assertIs(view[flag], False, flag)
        # Pale Carto basemap (positron, no key) under our own nearby place names.
        self.assertEqual(spec["mapProvider"], "carto")
        self.assertIn("positron-nolabels", spec["mapStyle"])
        places = next(l for l in spec["layers"] if l["id"] == "nearby-places")
        self.assertIn("Bellevue", [row["text"] for row in places["data"]])
        for row in places["data"]:
            self.assertFalse(self.boundary.contains(Point(row["lon"], row["lat"])), row["text"])
        areas = next(l for l in spec["layers"] if l.get("pickable"))
        self.assertFalse(areas["extruded"])
        self.assertEqual(areas["getLineColor"], [255, 255, 255, 255])
        self.assertEqual(areas["data"]["features"][0]["properties"]["fill"][3], 255)
        ids = [l["id"] for l in spec["layers"]]
        self.assertIn("picked-rim-halo", ids)
        self.assertIn("picked-rim", ids)
        self.assertIn("prediction-area-labels", ids)
        rim = next(l for l in spec["layers"] if l["id"] == "picked-rim")
        self.assertEqual(rim["data"][0]["path"][0][2], 0.0)
        label_layer = next(l for l in spec["layers"] if l["id"] == "prediction-area-labels")
        self.assertEqual(len(label_layer["data"]), 34)
        for name, row in zip(labels, label_layer["data"]):
            self.assertTrue(self.areas[name].contains(Point(row["lon"], row["lat"])), name)

    def test_forecast_map_without_fills_is_one_neutral_shade(self):
        # A date with no estimate: same layers, one pale sage, numbers only, a plain hover card.
        geoms, rings = area_geometries(self.areas)
        labels = area_labels(self.areas)
        deck = forecast_map(geoms, rings, self.boundary, None, labels)
        spec = json.loads(deck.to_json())
        areas = next(l for l in spec["layers"] if l.get("pickable"))
        fills = {tuple(f["properties"]["fill"]) for f in areas["data"]["features"]}
        self.assertEqual(fills, {(207, 214, 196, 255)})
        props = areas["data"]["features"][0]["properties"]
        self.assertEqual(props["status"], "No estimate for this shift")
        self.assertNotIn("expected", props)
        label_layer = next(l for l in spec["layers"] if l["id"] == "prediction-area-labels")
        self.assertTrue(all(row["text"].isdigit() for row in label_layer["data"]))
        self.assertNotIn("Expected", deck._tooltip["html"])
        self.assertIn("nearby-places", [l["id"] for l in spec["layers"]])


class StationAreaScreenTests(unittest.TestCase):
    def area_map(self, app):
        # The Stations map is the one with the clickable station-areas layer in the Explore tab
        # (Citywide and Events have their own maps, and the Prediction tab another).
        stations = next(t for t in app.tabs if t.label == "Stations")
        for chart in stations.get("deck_gl_json_chart"):
            spec = json.loads(chart.proto.json)
            if any(layer.get("id") == "station-areas" for layer in spec["layers"]):
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
        caption = " ".join(plain(h.value) for h in app.tabs[1].get("html"))
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
                "picked-outline": {"getLineColor": "@@=properties.line"},
            }.get(layer["id"], {})
            self.assertEqual(expressions, allowed, layer["@@type"])
            if layer["@@type"] == "GeoJsonLayer":
                self.assertLessEqual(layer["lineWidthMaxPixels"], 3)
            else:
                self.assertLessEqual(layer["radiusMaxPixels"], 6)

    def test_future_dates_use_the_last_known_week(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        app.date_input[0].set_value(date(2026, 12, 25)).run(timeout=60)
        self.assertFalse(app.exception)
        areas = next(l for l in self.area_map(app)["layers"] if l.get("pickable"))
        sfd10 = next(f["properties"] for f in areas["data"]["features"] if f["properties"]["Station"] == "SFD 10")
        self.assertEqual(sfd10["week_dates"], "Oct 22 to Oct 28")


if __name__ == "__main__":
    unittest.main()
