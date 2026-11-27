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
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
from shapely import MultiPoint, Point, Polygon, make_valid, transform, voronoi_polygons
from shapely.geometry import mapping, shape
from shapely.ops import polylabel

from chart_theme import (
    DAY_STOPS, FS_SM, FS_XS, INK, LIGHT_TEXT, SAGE, SAGE_DARK, SAGE_GREY, STATION_COLORS,
    STATION_EDGES, rgb,
)

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
# Every clickable map shades the area under the mouse the same way.
HOVER_COLOR = rgb(INK, 65)
LABEL_MIN_RADIUS_KM = 0.83  # smaller inner areas only fit the station number


def area_selection(event, layer_id=None):
    # Return (layer_id, clicked station names) from st.pydeck_chart's selection state.
    if event is None:
        return None, []
    objects_by_layer = event.get("selection", {}).get("objects", {})
    if layer_id is not None:
        if layer_id not in objects_by_layer:
            return None, []
        names = [o.get("properties", {}).get("Station") for o in objects_by_layer[layer_id]]
        return layer_id, list(dict.fromkeys(n for n in names if n))
    for key, objects in objects_by_layer.items():
        if key == AREA_LAYER_ID or key.startswith(f"{AREA_LAYER_ID}-"):
            names = [o.get("properties", {}).get("Station") for o in objects]
            return key, list(dict.fromkeys(n for n in names if n))
    return None, []


def selected_stations(event):
    # Read clicked station names from st.pydeck_chart's selection state, keeping order.
    return area_selection(event)[1]


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


# The darker edge for each pick's rim colour; any other colour keeps a white halo.
RIM_EDGES = {tuple(rgb(c)): rgb(e, 255) for c, e in zip(STATION_COLORS, STATION_EDGES)}
CLEAR_FILL = rgb(SAGE_GREY, 12)  # nearly clear: outlined, not coloured, but hoverable
# DAY_STOPS (chart_theme.py) run mint to rose: "more" is rose. The picked stations'
# lavender, yellow and mint outlines sit on a darker edge of their own hue, so they stay
# readable on top of it.
SHADE_ALPHA = 210
# One pale sage for every area on a date with no estimate: no forecast colours, dark numbers.
NO_ESTIMATE_FILL = rgb(SAGE, 255)


def shade(values, low=None, high=None, colors=DAY_STOPS):
    """RGBA fill per station: the lowest value gets the first colour, the highest the last."""
    values = pd.Series(values, dtype=float)
    low = values.min() if low is None else low
    high = values.max() if high is None else high
    stops = np.array([rgb(c) for c in colors], dtype=float)
    out = {}
    for name, value in values.items():
        t = 0.0 if high <= low or pd.isna(value) else float(np.clip((value - low) / (high - low), 0, 1))
        pos = t * (len(stops) - 1)
        i = min(int(pos), len(stops) - 2)
        mix = stops[i] + (stops[i + 1] - stops[i]) * (pos - i)
        out[name] = [int(round(c)) for c in mix] + [SHADE_ALPHA]
    return out


def _city_layer(boundary, layer_id, color=tuple(rgb(INK)), width=3):
    return pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(boundary), "properties": {}}]},
        id=layer_id,
        stroked=True,
        filled=False,
        get_line_color=list(color),
        # Widths are in metres; the min/max pixel limits keep the line fixed at any zoom.
        # (A units string such as "pixels" would reach the browser as an expression, not text.)
        get_line_width=1,
        line_width_min_pixels=width,
        line_width_max_pixels=width,
    )


TOOLTIP_STYLE = {
    "backgroundColor": INK,
    "color": LIGHT_TEXT,
    "fontSize": f"{FS_SM}px",
    "borderRadius": "8px",
    "padding": "8px 10px",
}
# The Prediction map cannot move: zoom and pitch are pinned (min = max), so they hold even when
# Streamlit swaps in a new view state. Streamlit 1.63 always turns pydeck's controller on; the
# switches below turn its mouse, touch and key moves off, and MAP_LOCK_JS keeps a drag from
# ever starting (a started drag would make Streamlit drop the switches). fixed_view adds the
# centre and zoom.
FIXED_VIEW = dict(
    pitch=0,
    min_pitch=0,
    max_pitch=0,
    bearing=0,
    drag_pan=False,
    drag_rotate=False,
    scroll_zoom=False,
    double_click_zoom=False,
    touch_zoom=False,
    touch_rotate=False,
    keyboard=False,
)
# The narrowest Prediction map is on a 390 px phone (360 px inside the frame). The city and the
# town names fit it with MAP_PAD px spare at the sides; the frame's height fits them with
# MAP_PAD_Y px above and below, and keeps at least CITY_PAD_Y px of land above and below the
# city. Wider frames just show more of the pale basemap.
MAP_MIN_WIDTH = 360
MAP_PAD = 6
MAP_PAD_Y = 8
CITY_PAD_Y = 40
# Stops drags, the wheel and keys on the Prediction map before deck.gl sees them. Hover (no
# button held) and clicks (press and release) still pass, and the wheel scrolls the page.
MAP_LOCK_JS = """<script>
if (!window.mapLocked) {
  window.mapLocked = true;
  const onMap = (e) => e.target instanceof Element && e.target.closest(".st-key-prediction_map_box");
  let held = false;
  addEventListener("pointerdown", (e) => { held = Boolean(onMap(e)); }, true);
  addEventListener("pointermove", (e) => { if (held && e.buttons) e.stopPropagation(); }, true);
  for (const t of ["pointerup", "pointercancel"]) addEventListener(t, () => { held = false; });
  for (const t of ["wheel", "keydown"]) {
    addEventListener(t, (e) => { if (onMap(e)) e.stopPropagation(); }, true);
  }
}
</script>"""
# Nearby towns drawn on the map itself (the basemap has none), at their real town centres;
# two-line names stay narrow and Renton sits a little north, clear of the info button. Vashon
# and Puget Sound sit too far west for the phone frame.
NEARBY_PLACES = [
    ("Shoreline", -122.341, 47.756),
    ("Lake Forest\nPark", -122.281, 47.757),
    ("Kirkland", -122.209, 47.682),
    ("Bellevue", -122.201, 47.61),
    ("Mercer\nIsland", -122.222, 47.571),
    ("Renton", -122.217, 47.49),
    ("Tukwila", -122.261, 47.474),
    ("Burien", -122.347, 47.47),
]


def mercator(lon, lat):
    # deck.gl's flat map at zoom 0: the world is 512 px across, y grows southward.
    y = math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
    return (lon + 180) / 360 * 512, (1 - y / math.pi) / 2 * 512


def label_size(name):
    # Rough size in px of a name drawn at FS_XS (14 px): about half the size a letter
    # ("Shoreline" draws 61 px wide) and 1.27 of it a line.
    lines = name.split("\n")
    return max(map(len, lines)) * 0.5 * FS_XS + 6, len(lines) * 1.27 * FS_XS + 2


def fixed_view(boundary):
    # Centre and zoom that fit the city and the town names into the narrowest frame, and that
    # frame's height. Each item is a point in zoom-0 px with the width and height it takes up.
    w, s, e, n = boundary.bounds
    items = [(*mercator(w, n), 0, 0), (*mercator(e, s), 0, 0)]
    items += [(*mercator(lon, lat), *label_size(t)) for t, lon, lat in NEARBY_PLACES]
    # Each pair of items fills the frame's width at one zoom; the smallest of those fits them all.
    room = MAP_MIN_WIDTH - 2 * MAP_PAD
    zoom = min(
        math.log2((room - (wa + wb) / 2) / (xa - xb))
        for xa, _, wa, _ in items
        for xb, _, wb, _ in items
        if xa > xb
    )
    k = 2**zoom
    left = min(x * k - wd / 2 for x, _, wd, _ in items)
    right = max(x * k + wd / 2 for x, _, wd, _ in items)
    top = min(min(y * k - ht / 2 for _, y, _, ht in items), items[0][1] * k - CITY_PAD_Y)
    bottom = max(max(y * k + ht / 2 for _, y, _, ht in items), items[1][1] * k + CITY_PAD_Y)
    height = math.ceil(bottom - top + 2 * MAP_PAD_Y)
    x, y = (left + right) / 2 / k, (top + bottom) / 2 / k
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - y / 256))))
    view = dict(latitude=lat, longitude=x / 512 * 360 - 180, zoom=zoom, min_zoom=zoom, max_zoom=zoom)
    return {**view, **FIXED_VIEW}, height


def value_map(areas, boundary, values, value_text, layer_id=AREA_LAYER_ID):
    """Explore > Citywide: each station area shaded by one value (e.g. fire calls per shift).

    value_text maps a station to its hover line. Clicking an area opens its history window.
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
        id=layer_id,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill",
        get_line_color=[255, 255, 255, 220],
        get_line_width=1,
        line_width_min_pixels=1,
        line_width_max_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=HOVER_COLOR,
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
    """A darker edge and a coloured outline around each picked station (its chart colour)."""
    features = [
        {"type": "Feature", "geometry": mapping(areas[name]),
         "properties": {"line": color, "edge": RIM_EDGES.get(tuple(color[:3]), [255, 255, 255, 255])}}
        for name, color in picked_colors.items() if name in areas
    ]
    data = {"type": "FeatureCollection", "features": features}
    halo = pdk.Layer("GeoJsonLayer", data, id="picked-halo", stroked=True, filled=False,
                     get_line_color="properties.edge", get_line_width=1,
                     line_width_min_pixels=7, line_width_max_pixels=7)
    outline = pdk.Layer("GeoJsonLayer", data, id="picked-outline", stroked=True, filled=False,
                        get_line_color="properties.line", get_line_width=1,
                        line_width_min_pixels=3.5, line_width_max_pixels=3.5)
    return [halo, outline]


def area_geometries(areas):
    # Cache GeoJSON geometry dicts and outer rings once so map reruns only swap properties.
    geoms = {name: mapping(polygon) for name, polygon in areas.items()}
    rings = {name: [list(pt) for pt in polygon.exterior.coords] for name, polygon in areas.items()}
    return geoms, rings


def area_labels(areas):
    """Interior label anchor inside each polygon, plus whether expected calls fit too."""
    out = {}
    for name, polygon in areas.items():
        poly_km = transform(polygon, to_km)
        pt = polylabel(poly_km, tolerance=0.01)
        lon, lat = to_degrees(np.array([pt.x, pt.y]))
        out[name] = {
            "lon": float(lon),
            "lat": float(lat),
            "num": name.split()[-1],
            "fits_count": poly_km.boundary.distance(pt) >= LABEL_MIN_RADIUS_KM,
        }
    return out


def _picked_rim_layers(rings, picked_colors):
    # Darker edge and station-colour rim around each picked area so rims stay crisp on any fill.
    paths = [
        {"path": [[pt[0], pt[1], 0.0] for pt in rings[name]], "color": color,
         "edge": RIM_EDGES.get(tuple(color[:3]), [255, 255, 255, 255])}
        for name, color in picked_colors.items()
        if name in rings
    ]
    halo = pdk.Layer(
        "PathLayer",
        paths,
        id="picked-rim-halo",
        get_path="path",
        get_color="edge",
        width_min_pixels=7,
        width_max_pixels=7,
    )
    outline = pdk.Layer(
        "PathLayer",
        paths,
        id="picked-rim",
        get_path="path",
        get_color="color",
        width_min_pixels=3.5,
        width_max_pixels=3.5,
    )
    return [halo, outline]


def forecast_map(
    geoms,
    rings,
    boundary,
    fills,
    labels,
    expected=None,
    picked_colors=None,
    details=None,
    layer_id=AREA_LAYER_ID,
):
    # Flat, north-up Prediction map on a pale basemap; colour shows expected vs usual.
    # With no fills (a date with no estimate) every area gets the same pale sage instead.
    neutral = fills is None
    if neutral:
        fills = dict.fromkeys(geoms, NO_ESTIMATE_FILL)
        details = {name: {"status": "No estimate for this shift"} for name in geoms}
    features = []
    for name, geom in geoms.items():
        props = {
            "Station": name,
            "fill": [*fills[name][:3], 255],
            **(details or {}).get(name, {}),
        }
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    area_layer = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": features},
        id=layer_id,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=False,
        get_fill_color="properties.fill",
        get_line_color=[255, 255, 255, 255],
        get_line_width=1,
        line_width_min_pixels=1.5,
        line_width_max_pixels=2,
        pickable=True,
        auto_highlight=True,
        highlight_color=HOVER_COLOR,
    )
    text_rows = []
    for name, info in labels.items():
        r, g, b = fills[name][:3]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        # Below about 132 white text has more contrast than dark text; above it, dark wins.
        color = [255, 255, 255, 245] if luma < 132 else rgb(INK, 240)
        if expected is not None and info["fits_count"]:
            text = f"{info['num']}\n{float(expected[name]):.1f}"
        else:
            text = info["num"]
        text_rows.append({"lon": info["lon"], "lat": info["lat"], "text": text, "color": color})
    label_layer = pdk.Layer(
        "TextLayer",
        text_rows,
        id="prediction-area-labels",
        get_position=["lon", "lat"],
        get_text="text",
        get_color="color",
        get_size=FS_XS,
        font_family="'Figtree, Helvetica Neue, Arial, sans-serif'",
        font_weight=600,
        pickable=False,
    )
    # A white wash over the basemap keeps it pale; the opaque areas sit on top of it.
    wash = pdk.Layer(
        "PolygonLayer",
        [{"ring": [[-123.2, 47.1], [-121.5, 47.1], [-121.5, 48.1], [-123.2, 48.1]]}],
        id="basemap-wash",
        get_polygon="ring",
        get_fill_color=[255, 255, 255, 120],
        stroked=False,
        pickable=False,
    )
    place_layer = pdk.Layer(
        "TextLayer",
        [{"text": t, "lon": lon, "lat": lat} for t, lon, lat in NEARBY_PLACES],
        id="nearby-places",
        get_position=["lon", "lat"],
        get_text="text",
        get_color=rgb(SAGE_GREY, 230),
        get_size=FS_XS,
        font_family="'Figtree, Helvetica Neue, Arial, sans-serif'",
        font_weight=500,
        # A white halo keeps the names readable over roads and water.
        font_settings={"sdf": True},
        outline_width=4,
        outline_color=[255, 255, 255, 230],
        pickable=False,
    )
    layers = [
        wash,
        place_layer,
        area_layer,
        _city_layer(boundary, "prediction-city-boundary", color=rgb(SAGE_DARK, 210), width=1.5),
    ]
    if picked_colors:
        layers.extend(_picked_rim_layers(rings, picked_colors))
    layers.append(label_layer)
    tooltip = {
        "html": (
            "<b>{Station}</b> · {status}<br/>"
            "Expected {expected} calls ({diff} vs usual)<br/>"
            "Likely range {range}<br/>"
            "Usual {usual}"
            if not neutral
            else "<b>{Station}</b><br/>{status}"
        ),
        "style": {**TOOLTIP_STYLE, "maxWidth": "185px", "whiteSpace": "normal"},
    }
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**fixed_view(boundary)[0]),
        # Pale Carto basemap without labels (free, no key); the opaque areas cover it inside
        # Seattle. With the tiles offline the areas and place names still draw on the box colour.
        map_provider="carto",
        map_style="light_no_labels",
        tooltip=tooltip,
    )


def area_map(areas, boundary, stations, averages, fills=None, picked_colors=None, extra=None,
             details=None, tooltip_html=None, layer_id=AREA_LAYER_ID):
    """pydeck map: city outline, station areas, station dots, hover.

    fills shades the areas (e.g. by fire share; nearly clear if not given). Picked stations get
    a coloured outline (picked_colors: station -> RGBA). extra adds a hover line per station;
    details and tooltip_html replace the whole hover text (the tooltip fills {names} from
    each station's details).
    """
    # Rotating layer_id clears Streamlit's clicked fill highlight after each map click.
    city_layer = _city_layer(boundary, "city-boundary")
    area_layer = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": area_features(areas, averages, fills, extra, details)},
        id=layer_id,
        stroked=True,
        filled=True,
        # Each feature carries its own fill: its shade, or nearly clear (so the whole area,
        # not just its edge, reacts to the mouse).
        get_fill_color="properties.fill",
        get_line_color=[255, 255, 255, 220] if fills else rgb(SAGE_GREY, 200),
        get_line_width=1,
        line_width_min_pixels=1,
        line_width_max_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=HOVER_COLOR,
    )
    dot_layer = pdk.Layer(
        "ScatterplotLayer",
        stations[["station", "lon", "lat"]],
        id="station-dots",
        get_position=["lon", "lat"],
        get_fill_color=rgb(INK, 200),  # dark ink, so the shading and outlines stand out
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
