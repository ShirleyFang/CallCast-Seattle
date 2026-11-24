"""Explore > Events: how calls changed at the stations next to each venue on past event shifts.

Reads data/event_effects.csv, made by scripts/make_event_effects.py (2022-2024, the same
method and years as the event analysis in scripts/evaluate_events.py).
"""

from pathlib import Path

import pandas as pd
import pydeck as pdk
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parent
EFFECTS_PATH = ROOT / "data/event_effects.csv"
TABLE_NAMES = {
    "label": "Event",
    "venue": "Venue",
    "station": "Station",
    "events": "Events",
    "usual_calls": "Usual calls",
    "event_calls": "Calls on event days",
    "extra_calls": "Extra calls",
    "extra_pct": "Extra %",
}


def load_event_effects(path=EFFECTS_PATH):
    effects = pd.read_csv(path)
    needed = {"label", "venue", "station", "events", "usual_calls", "event_calls",
              "extra_calls", "extra_pct", "clear", "years", "venue_lat", "venue_lon"}
    if needed - set(effects.columns):
        raise ValueError("event_effects.csv is missing columns; run scripts/make_event_effects.py.")
    return effects


# Stations with a clear event effect, one colour each (fill, outline): gold for SFD 10
# (Lumen Field / T-Mobile Park), green for SFD 22 (Husky Stadium); others fall back to gold.
AFFECTED_COLORS = {
    "SFD 10": ([201, 154, 59, 150], [150, 110, 30]),
    "SFD 22": ([94, 125, 90, 150], [60, 85, 57]),
}
DEFAULT_AFFECTED = ([201, 154, 59, 150], [150, 110, 30])
VENUE_COLOR = [31, 42, 54]
# Lumen Field and T-Mobile Park are about 400 m apart: put one label above, one below.
LABEL_OFFSETS = {"T-Mobile Park": [0, 16]}


def _signed(x):
    return f"{x:+.1f}"


def venue_rows(effects):
    """One row per venue: position, label offset, and hover text listing its events."""
    rows = []
    for venue, group in effects.groupby("venue"):
        clear = group[group["clear"]]
        lines = [
            f"{r.label} → {r.station}: {_signed(r.extra_calls)} calls per event day ({r.events} events)"
            for r in clear.itertuples()
        ]
        others = sorted(set(group.loc[~group["clear"], "label"]) - set(clear["label"]))
        if others:
            lines.append("No clear effect: " + ", ".join(others))
        rows.append({
            "venue": venue,
            "lat": group["venue_lat"].iloc[0],
            "lon": group["venue_lon"].iloc[0],
            "offset": LABEL_OFFSETS.get(venue, [0, -16]),
            "title": venue,
            "body": "\n".join(lines),  # plain-text tooltip: newlines, not <br/>, break lines
        })
    return pd.DataFrame(rows)


def affected_features(effects, areas):
    """GeoJSON for the station areas with a clear effect, with hover text."""
    features = []
    clear = effects[effects["clear"]]
    for station, group in clear.groupby("station"):
        body = "\n".join(
            f"{r.label} at {r.venue}: {_signed(r.extra_calls)} calls per event day"
            for r in group.itertuples()
        )
        features.append({
            "type": "Feature",
            "geometry": mapping(areas[station]),
            "properties": {
                "title": station,
                "body": body,
                "fill": AFFECTED_COLORS.get(station, DEFAULT_AFFECTED)[0],
                "line": AFFECTED_COLORS.get(station, DEFAULT_AFFECTED)[1],
            },
        })
    return features


def venue_map(effects, areas, boundary):
    """City outline, all station areas outlined, clear-effect areas coloured, venues labelled."""
    outlines = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection",
         "features": [{"type": "Feature", "geometry": mapping(a), "properties": {}} for a in areas.values()]},
        id="event-station-areas",
        stroked=True, filled=False,
        get_line_color=[47, 93, 138, 160], get_line_width=1,
        line_width_min_pixels=1, line_width_max_pixels=1,
    )
    affected = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": affected_features(effects, areas)},
        id="event-affected-areas",
        stroked=True, filled=True,
        # each area carries its own colour in its properties (see AFFECTED_COLORS)
        get_fill_color="properties.fill", get_line_color="properties.line",
        get_line_width=1, line_width_min_pixels=2, line_width_max_pixels=2,
        pickable=True,
    )
    city = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(boundary), "properties": {}}]},
        id="event-city-boundary",
        stroked=True, filled=False, get_line_color=[31, 42, 54],
        get_line_width=1, line_width_min_pixels=3, line_width_max_pixels=3,
    )
    venues = venue_rows(effects)
    dots = pdk.Layer(
        "ScatterplotLayer", venues, id="event-venues",
        get_position=["lon", "lat"], get_fill_color=VENUE_COLOR,
        get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=2,
        get_radius=60, radius_min_pixels=6, radius_max_pixels=9,
        pickable=True,
    )
    labels = pdk.Layer(
        "TextLayer", venues, id="event-venue-labels",
        get_position=["lon", "lat"], get_text="venue", get_pixel_offset="offset",
        get_size=13, get_color=VENUE_COLOR,  # centred on the dot by default
        # The app's font. pydeck keeps a string as text only if it starts and ends with the
        # same quote (which it strips); any other string becomes a JavaScript expression.
        font_family="'Public Sans, Helvetica Neue, Arial, sans-serif'",
        font_weight=600, background=True, get_background_color=[250, 248, 245, 220],
    )
    tooltip = {
        # Plain text: the map escapes HTML inside values, so <br/> would show as text.
        "text": "{title}\n{body}",
        "style": {"backgroundColor": "#1F2A36", "color": "white", "fontSize": "13px",
                  "maxWidth": "340px", "lineHeight": "1.45"},
    }
    return pdk.Deck(
        layers=[outlines, affected, city, dots, labels],
        # Zoomed so all four venues fit with room below Lumen Field for its hover text.
        initial_view_state=pdk.ViewState(latitude=47.623, longitude=-122.33, zoom=11.1),
        map_provider="carto",
        map_style="light",
        tooltip=tooltip,
    )


def effects_table(effects):
    """Rows to show, in plain words: rounded calls, ordered by the size of the change."""
    table = effects.sort_values("extra_pct", ascending=False)[list(TABLE_NAMES)]
    table = table.round({"usual_calls": 1, "event_calls": 1, "extra_calls": 1, "extra_pct": 0})
    return table.rename(columns=TABLE_NAMES)


def _show_table(st, rows):
    """One effects table: one decimal everywhere, signed changes, every row visible."""
    one_decimal = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(
        effects_table(rows),
        hide_index=True,
        width="stretch",
        height=38 + 35 * len(rows),  # header + rows, so no row hides behind a scrollbar
        column_config={
            "Usual calls": one_decimal,
            "Calls on event days": one_decimal,
            "Extra calls": st.column_config.NumberColumn(format="%+.1f"),
            "Extra %": st.column_config.NumberColumn(format="%+.0f%%"),
        },
    )


def show_event_effects(st, areas=None, boundary=None, path=EFFECTS_PATH):
    """Draw the Events view into a Streamlit container (the map only if areas are given)."""
    try:
        effects = load_event_effects(path)
    except (OSError, ValueError) as exc:
        st.info(f"Past event effects are not available: {exc}")
        return
    years = effects["years"].iloc[0]
    first, last = years.split("-")
    # Unlike Citywide and Stations, these numbers are fixed: they do not follow the date.
    st.caption(
        f"Past event days from Jan {first} to Dec {last}. These numbers do not change with "
        "the selected date."
    )
    if areas is not None and boundary is not None:
        st.write("Venues and the stations they affect")
        st.pydeck_chart(venue_map(effects, areas, boundary))
        st.caption(
            "Dark dots are the four large venues. Coloured areas are the stations where calls "
            "were clearly higher on event days: gold for SFD 10 (Lumen Field, T-Mobile Park), "
            "green for SFD 22 (Husky Stadium). Point at a venue or a coloured area for details."
        )
    st.write("How calls changed near the venues on past event days")
    st.caption(
        "For each venue, the 2 nearest stations. Usual calls = the same station and weekday, "
        "up to 4 weeks before and after, without that kind of event. "
        f"Based on {years}."
    )
    clear = effects[effects["clear"]]
    st.write("Clear effect")
    _show_table(st, clear)
    chosen_on = effects["chosen_on"].iloc[0] if "chosen_on" in effects else years
    st.caption(
        "On these event days the station usually had more calls than on similar days: at "
        "least 10 events, at least 10% more, and consistent from event to event (pairs "
        f"chosen on {chosen_on}). These are averages: some event days were much busier and "
        "some were not busier at all."
    )
    st.write("No clear effect")
    _show_table(st, effects[~effects["clear"]])
    st.caption(
        "Smaller or less consistent changes. Arena events at Climate Pledge Arena and most "
        "games at T-Mobile Park did not change calls much at the nearest stations."
    )
