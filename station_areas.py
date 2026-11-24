"""Station area map for the Explore tab: the Seattle city boundary and each station's area.

A station's area is the part of Seattle closer to that station than to any other one. This is
the same rule notebooks/data_clean.ipynb uses to give each call to a station (its nearest
station), so the map and the counts agree. Real first-due areas follow roads and bridges, so
the edges are approximate.

Hovering over an area shows its average calls per shift over the last 7 and 28 shifts known at
the planning time (the same finished, gap-free shifts the rest of the app uses).

Run from the repository root to save the areas for review or the report:
    uv run python station_areas.py        # writes data/station_areas.geojson
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
from shapely import MultiPoint, Point, Polygon, make_valid, transform, voronoi_polygons
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parent
BOUNDARY_PATH = ROOT / "data/seattle_area.geojson"
AREAS_PATH = ROOT / "data/station_areas.geojson"
# Degrees are not the same length east-west and north-south, so nearest-station areas are
# drawn on a flat map centred on Seattle (km), where distances match the haversine distance
# the notebook uses to within a fraction of a percent at city scale.
LAT0, LON0 = 47.61, -122.33
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(LAT0))
RECENT_WINDOWS = {"week": 7, "month": 28}  # shifts averaged for the hover text
AREA_LAYER_ID = "station-areas"  # the clickable layer; the app reads the selection from it


def selected_stations(event):
    """Station names of the areas clicked on the map, from st.pydeck_chart's selection state."""
    if event is None:
        return []
    objects = event.get("selection", {}).get("objects", {}).get(AREA_LAYER_ID, [])
    names = [o.get("properties", {}).get("Station") for o in objects]
    return list(dict.fromkeys(n for n in names if n))  # keep order, drop repeats


def to_km(coords):
    """(lon, lat) rows -> (x, y) km on the flat map."""
    return (coords - [LON0, LAT0]) * [KM_PER_DEG_LON, KM_PER_DEG_LAT]


def to_degrees(coords):
    """(x, y) km rows on the flat map -> (lon, lat)."""
    return coords / [KM_PER_DEG_LON, KM_PER_DEG_LAT] + [LON0, LAT0]


def load_boundary(path=BOUNDARY_PATH):
    """The City of Seattle polygon (lon/lat), cleaned of a drawing error in the source file.

    The source ring starts on the Magnolia shore with a 200 m spike that folds back on
    itself (vertices 0-3, a 44 m2 sliver), so the raw polygon is invalid and the spike
    shows as a stray line near SFD 41. make_valid turns the sliver into a hole; keeping
    only the outer ring removes it. The city area is unchanged.
    """
    features = json.loads(Path(path).read_text())["features"]
    if len(features) != 1:
        raise ValueError("seattle_area.geojson should hold one city polygon.")
    fixed = make_valid(shape(features[0]["geometry"]))
    parts = [g for g in getattr(fixed, "geoms", [fixed]) if g.geom_type == "Polygon"]
    city = max(parts, key=lambda g: g.area)  # the city itself, not leftover slivers
    return Polygon(city.exterior)


def build_station_areas(stations, boundary):
    """Each station's nearest-station area inside the city, as {station: polygon (lon/lat)}."""
    city = transform(boundary, to_km)
    xy = to_km(stations[["lon", "lat"]].to_numpy(float))
    points = [Point(p) for p in xy]
    cells = voronoi_polygons(MultiPoint(points), extend_to=city.envelope.buffer(10)).geoms
    areas = {}
    for name, point in zip(stations["station"], points):
        cell = next(c for c in cells if c.contains(point))  # one Voronoi cell per station
        areas[name] = transform(cell.intersection(city), to_degrees)
    return areas


def area_km2(polygon):
    """Area of a lon/lat polygon in square km."""
    return transform(polygon, to_km).area


def recent_averages(history, windows=RECENT_WINDOWS):
    """Average calls per shift for each station over the last n known shifts, plus the dates.

    history is the app's known_history(): finished shifts by the planning time, no gaps.
    The windows count calendar shifts back from the latest known one; a missing-data shift
    inside a window is skipped, not counted as zero.
    """
    latest = history["ShiftDate"].max()
    out = {}
    for name, n in windows.items():
        start = latest - pd.Timedelta(days=n - 1)
        part = history[history["ShiftDate"] >= start]
        out[name] = {
            "mean": part.groupby("Station")["calls"].mean(),
            "from": start,
            "to": latest,
        }
    return out


CLEAR_FILL = [47, 93, 138, 12]  # nearly clear: outlined, not coloured, but hoverable


def area_features(areas, averages=None, fills=None):
    """GeoJSON features for the station areas, with the hover text in their properties.

    fills maps a station to an RGBA fill (picked stations get their chart colour).
    """
    features = []
    for name, polygon in areas.items():
        props = {"Station": name, "fill": (fills or {}).get(name, CLEAR_FILL)}
        if averages:
            for key, window in averages.items():
                value = window["mean"].get(name, np.nan)
                props[f"{key}_avg"] = "n/a" if pd.isna(value) else f"{value:.1f}"
                props[f"{key}_dates"] = f"{window['from']:%b %d} to {window['to']:%b %d}"
        features.append({"type": "Feature", "geometry": mapping(polygon), "properties": props})
    return features


def area_map(areas, boundary, stations, averages, fills=None):
    """pydeck map: city outline, station area outlines, station dots, hover.

    Areas are outlined only, except the picked ones, which take the colour in fills.
    """
    # Fixed layer ids: Streamlit needs them to keep the clicked areas selected across reruns.
    city_layer = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(boundary), "properties": {}}]},
        id="city-boundary",
        stroked=True,
        filled=False,
        get_line_color=[31, 42, 54],
        # Widths are in metres; the min/max pixel limits keep the line about 3 px at any zoom.
        # (A units string such as "pixels" would reach the browser as an expression, not text.)
        get_line_width=1,
        line_width_min_pixels=3,
        line_width_max_pixels=3,
    )
    area_layer = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": area_features(areas, averages, fills)},
        id=AREA_LAYER_ID,
        stroked=True,
        filled=True,
        # Each feature carries its own fill: nearly clear (so the whole area, not just its
        # edge, reacts to the mouse), or the chart colour of a picked station.
        get_fill_color="properties.fill",
        get_line_color=[47, 93, 138, 200],
        get_line_width=1,
        line_width_min_pixels=1,
        line_width_max_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[47, 93, 138, 60],
    )
    dot_layer = pdk.Layer(
        "ScatterplotLayer",
        stations[["station", "lon", "lat"]],
        id="station-dots",
        get_position=["lon", "lat"],
        get_fill_color=[164, 40, 43, 220],
        get_radius=80,  # metres, kept between 3 and 6 px on screen
        radius_min_pixels=3,
        radius_max_pixels=6,
    )
    tooltip = {
        "html": "<b>{Station}</b><br/>"
        "Last week: {week_avg} calls per shift<br/><small>{week_dates}</small><br/>"
        "Last month: {month_avg} calls per shift<br/><small>{month_dates}</small>",
        "style": {"backgroundColor": "#1F2A36", "color": "white", "fontSize": "13px"},
    }
    return pdk.Deck(
        layers=[area_layer, city_layer, dot_layer],
        initial_view_state=pdk.ViewState(latitude=47.61, longitude=-122.335, zoom=10.2),
        map_provider="carto",
        map_style="light",
        tooltip=tooltip,
    )


def main():
    stations = pd.read_csv(ROOT / "data/sfd_stations.csv")
    boundary = load_boundary()
    areas = build_station_areas(stations, boundary)
    collection = {"type": "FeatureCollection", "features": area_features(areas)}
    AREAS_PATH.write_text(json.dumps(collection))
    total = sum(area_km2(a) for a in areas.values())
    print(f"saved {AREAS_PATH.relative_to(ROOT)}: {len(areas)} station areas, "
          f"{total:.1f} km2 (city {area_km2(boundary):.1f} km2)")


if __name__ == "__main__":
    main()
