"""Station area map for the Prediction and Explore tabs: the Seattle city boundary and each station's area.

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
    # Read clicked station names from st.pydeck_chart's selection state, keeping order.
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
# Light grey to dark slate: "more" is darker. Slate is the app's neutral colour, so the picked
# stations' blue, red and gold outlines stay readable on top of it.
SHADE_STOPS = ["#F3F4F6", "#C9D1DA", "#8C97A4", "#4B5B6C", "#1F2A36"]
SHADE_ALPHA = 210


def _rgb(hex_color):
    return [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]


def shade(values, low=None, high=None, colors=SHADE_STOPS):
    """RGBA fill per station: the lowest value gets the first colour, the highest the last."""
    values = pd.Series(values, dtype=float)
    low = values.min() if low is None else low
    high = values.max() if high is None else high
    stops = np.array([_rgb(c) for c in colors], dtype=float)
    out = {}
    for name, value in values.items():
        t = 0.0 if high <= low or pd.isna(value) else float(np.clip((value - low) / (high - low), 0, 1))
        pos = t * (len(stops) - 1)
        i = min(int(pos), len(stops) - 2)
        rgb = stops[i] + (stops[i + 1] - stops[i]) * (pos - i)
        out[name] = [int(round(c)) for c in rgb] + [SHADE_ALPHA]
    return out


def _city_layer(boundary, layer_id):
    return pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(boundary), "properties": {}}]},
        id=layer_id,
        stroked=True,
        filled=False,
        get_line_color=[31, 42, 54],
        # Widths are in metres; the min/max pixel limits keep the line about 3 px at any zoom.
        # (A units string such as "pixels" would reach the browser as an expression, not text.)
        get_line_width=1,
        line_width_min_pixels=3,
        line_width_max_pixels=3,
    )


TOOLTIP_STYLE = {"backgroundColor": "#1F2A36", "color": "white", "fontSize": "13px"}


def value_map(areas, boundary, values, value_text):
    """Explore > Citywide: each station area shaded by one value (e.g. fire calls per shift).

    value_text maps a station to its hover line. View only: nothing can be clicked.
    """
    fills = shade(values)
    features = [
        {"type": "Feature", "geometry": mapping(polygon),
         "properties": {"Station": name, "fill": fills[name], "value": value_text.get(name, "n/a")}}
        for name, polygon in areas.items()
    ]
    layer = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": features},
        id="city-value-areas",
        stroked=True,
        filled=True,
        get_fill_color="properties.fill",
        get_line_color=[255, 255, 255, 220],
        get_line_width=1,
        line_width_min_pixels=1,
        line_width_max_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[164, 40, 43, 90],
    )
    return pdk.Deck(
        layers=[layer, _city_layer(boundary, "city-value-boundary")],
        initial_view_state=pdk.ViewState(latitude=47.61, longitude=-122.335, zoom=10.2),
        map_provider="carto",
        map_style="light",
        tooltip={"html": "<b>{Station}</b><br/>{value}", "style": TOOLTIP_STYLE},
    )


def area_features(areas, averages=None, fills=None, extra=None, details=None):
    """GeoJSON features for the station areas, with the hover text in their properties.

    fills maps a station to an RGBA fill (e.g. its share shade); extra maps a station to one
    more hover line (e.g. "Fire share: 27%"); details maps a station to a dict of further
    properties for a custom tooltip.
    """
    features = []
    for name, polygon in areas.items():
        props = {"Station": name, "fill": (fills or {}).get(name, CLEAR_FILL),
                 "extra": (extra or {}).get(name, ""), **(details or {}).get(name, {})}
        if averages:
            for key, window in averages.items():
                value = window["mean"].get(name, np.nan)
                props[f"{key}_avg"] = "n/a" if pd.isna(value) else f"{value:.1f}"
                props[f"{key}_dates"] = f"{window['from']:%b %d} to {window['to']:%b %d}"
        features.append({"type": "Feature", "geometry": mapping(polygon), "properties": props})
    return features


def picked_outline_layers(areas, picked_colors):
    """A white halo and a coloured outline around each picked station (its chart colour)."""
    features = [
        {"type": "Feature", "geometry": mapping(areas[name]), "properties": {"line": color}}
        for name, color in picked_colors.items() if name in areas
    ]
    data = {"type": "FeatureCollection", "features": features}
    halo = pdk.Layer("GeoJsonLayer", data, id="picked-halo", stroked=True, filled=False,
                     get_line_color=[255, 255, 255, 255], get_line_width=1,
                     line_width_min_pixels=7, line_width_max_pixels=7)
    outline = pdk.Layer("GeoJsonLayer", data, id="picked-outline", stroked=True, filled=False,
                        get_line_color="properties.line", get_line_width=1,
                        line_width_min_pixels=3.5, line_width_max_pixels=3.5)
    return [halo, outline]


def area_map(areas, boundary, stations, averages, fills=None, picked_colors=None, extra=None,
             details=None, tooltip_html=None):
    """pydeck map: city outline, station areas, station dots, hover.

    fills shades the areas (e.g. by fire share; nearly clear if not given). Picked stations get
    a coloured outline (picked_colors: station -> RGBA). extra adds a hover line per station;
    details and tooltip_html replace the whole hover text (the tooltip fills {names} from
    each station's details).
    """
    # Fixed layer ids: Streamlit needs them to keep the clicked areas selected across reruns.
    city_layer = _city_layer(boundary, "city-boundary")
    area_layer = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": area_features(areas, averages, fills, extra, details)},
        id=AREA_LAYER_ID,
        stroked=True,
        filled=True,
        # Each feature carries its own fill: its shade, or nearly clear (so the whole area,
        # not just its edge, reacts to the mouse).
        get_fill_color="properties.fill",
        get_line_color=[255, 255, 255, 220] if fills else [47, 93, 138, 200],
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
        get_fill_color=[31, 42, 54, 200],  # dark grey, so the shading and outlines stand out
        get_radius=80,  # metres, kept between 3 and 6 px on screen
        radius_min_pixels=3,
        radius_max_pixels=6,
    )
    tooltip = {
        "html": tooltip_html
        or "<b>{Station}</b><br/>{extra}<br/>"
        "Last week: {week_avg} calls per shift<br/><small>{week_dates}</small><br/>"
        "Last month: {month_avg} calls per shift<br/><small>{month_dates}</small>",
        "style": TOOLTIP_STYLE,
    }
    outlines = picked_outline_layers(areas, picked_colors) if picked_colors else []
    return pdk.Deck(
        layers=[area_layer, city_layer, *outlines, dot_layer],
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
