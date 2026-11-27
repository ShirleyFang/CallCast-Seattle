"""Build the CallCast Seattle page. Run with: uv run streamlit run app.py"""

import pandas as pd
import streamlit as st

from chart_theme import (
    PLOT_CONFIG,
    call_mix_chart,
    css_variables,
    history_line_chart,
    history_window_chart,
    hourly_altair_chart,
    scale_top,
)

from app_data import (
    GAP_MESSAGE,
    ROOT,
    add_likely_range,
    citywide_hourly,
    citywide_range,
    first_model_date,
    forecast_origin,
    hourly_allocation,
    known_history,
    learned_predictions,
    load_ranges,
    load_tables,
    reference_predictions,
    station_hour_peaks,
    target_dates,
    validate_predictions,
)
from prediction_cards import (
    briefing_html,
    history_legend_html,
    map_legend_html,
    TABLE_CSS,
    TABLE_JS,
    range_bar_html,
    ranked_table_html,
)
from station_areas import (
    AREA_LAYER_ID,
    MAP_LOCK_JS,
    area_geometries,
    area_labels,
    area_map,
    area_selection,
    build_station_areas,
    fixed_view,
    forecast_map,
    load_boundary,
    recent_averages,
    shade,
    value_map,
)
from event_explore import show_event_effects
from event_warnings import get_event_warnings
from weather_alerts import NOTE as WEATHER_NOTE
from weather_alerts import NWS_NOTE, get_weather_alerts, show_nws
from weather_now import alert_icon_css, get_shift_weather, nws_note_html, weather_now_html
from weather_explore import show_weather_effects
from ui_sounds import add_ui_sounds
import text_blocks as tb
from station_forecast import (
    BRIEF_USUAL_CALLS,
    HISTORY_SPANS,
    WATCH_MAX,
    WATCH_MIN_EXTRA,
    WATCH_MIN_SHARE,
    biggest_move,
    data_through,
    difference_fills,
    download_rows,
    history_summary,
    last_weeks,
    mix_table,
    range_bar_spec,
    ranked_rows,
    recent_days,
    same_weekday_calls,
    season_mix,
    shift_briefing,
    station_status_words,
    stations_to_watch,
    table_action,
    toggle_picks,
)
from station_compare import (
    CALL_TYPES,
    CITY,
    SHARE_TYPES,
    CITY_COLOR,
    MAX_COMPARE,
    TREND_DAYS,
    category_chart,
    category_means,
    category_shares,
    citywide,
    monthly,
    monthly_chart,
    monthly_means,
    only_type,
    station_bar_chart,
    station_color,
    station_means,
    station_shares,
    stacked_trend_chart,
    type_trend,
    summary as compare_summary,
    trend,
    trend_chart,
    weekday_chart,
    weekday_means,
)

# The pick-number badge in a compare card's top right corner.
PICK_BADGE = (
    "position: absolute; top: 10px; right: 12px; width: 26px; height: 26px; z-index: 1; "
    "border-radius: 50%; font-size: var(--fs-xs); font-weight: 700; line-height: 26px; text-align: center;"
)

# Session keys: the picked stations (the multiselect) and the Prediction map.
PICK_KEY = "compare_stations"
MAP_KEY = "prediction_map"
# The map's clicked areas at the last run, to tell which area a new click changed.
CLICKED_KEY = "map_clicked"
PICK_GEN_KEY = "prediction_map_gen"
EXPLORE_PICK_KEY = "explore_stations"
EXPLORE_MAP_KEY = "station_area_map"
EXPLORE_CLICKED_KEY = "explore_map_clicked"
EXPLORE_GEN_KEY = "explore_map_gen"
# Explore > Citywide map: a click opens that station's history window, read once per click.
CITY_MAP_KEY = "city_value_map"
CITY_GEN_KEY = "city_map_gen"
# The ranked table. Its click comes back for one run only, so an old click never reopens a window.
TABLE_KEY = "ranked_table"
# The time span picked inside the history window.
SPAN_KEY = "history_span"

# The first 3 shifts after the last day of counts use real recent calls (lag3 is known);
# later shifts use typical values instead (MIN_LAG and data/lag_fill_means.csv in step 4).
RECENT_CALL_SHIFTS = 3

# Outside pages a planner can open for live incidents and for station coverage (link chips).
LINKS = [
    ("Live incidents", "SFD Live", "https://sfdlive.com/"),
    (
        "Station coverage",
        "Seattle Breaking guide",
        "https://www.seattlebreaking.com/articles/seattle-fire-stations-coverage",
    ),
]

st.set_page_config(
    page_title="CallCast Seattle",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Hover colors, top bar and responsive layout live in style.css; its colours come from
# chart_theme.py as CSS variables. style.css goes first: its Lexend @import must lead.
st.html(f"<style>{(ROOT / 'style.css').read_text()}{css_variables()}</style>")
# Keeps the Prediction map still: drags and the wheel never reach it.
st.html(MAP_LOCK_JS, unsafe_allow_javascript=True)
# Hover tick, click and window-close sounds for everything clickable (ui_sounds.py).
add_ui_sounds()
# The station table. Registered on every run: Streamlit keeps one copy and starts each test
# app empty.
ranked_table = st.components.v2.component(
    "ranked_table", html='<div class="wrap"></div>', css=TABLE_CSS, js=TABLE_JS
)


def sync_map_picks(state, map_key, pick_key, clicked_key, gen_key):
    # Read clicks from the map's current layer and rotate its id so Streamlit clears the
    # clicked fill highlight before the next run (otherwise a refused click stays coloured).
    gen, prev_layer = state.get(gen_key, (0, None))
    current = AREA_LAYER_ID if gen == 0 else f"{AREA_LAYER_ID}-{gen}"
    event = state.get(map_key)
    layer, on_map = area_selection(event, current)
    if not layer and prev_layer:
        layer, on_map = area_selection(event, prev_layer)
    prev = state.get(clicked_key, []) if layer and layer == prev_layer else []
    # Sorted so two clicks that land in one run always give the same result.
    clicked = sorted(set(on_map) ^ set(prev)) if layer else []
    state[clicked_key] = on_map
    refused = []
    picks = list(state.get(pick_key, []))
    if clicked:
        picks, refused = toggle_picks(picks, clicked)
    # Saved on every run, before the compare list is drawn, so a map click reaches the list
    # (and the picks stay if the list is ever not drawn).
    state[pick_key] = picks
    if on_map and layer == current:
        gen += 1
    state[gen_key] = (gen, layer)
    next_layer = AREA_LAYER_ID if gen == 0 else f"{AREA_LAYER_ID}-{gen}"
    return picks, refused, next_layer


def refusal_text(refused):
    verb = "was" if len(refused) == 1 else "were"
    return (
        f"Up to {MAX_COMPARE} stations can be compared at once, so {', '.join(refused)} {verb} "
        "not added. Click a picked area or table row again, or remove it from the compare list, "
        "to make room."
    )


@st.cache_data
def read_data(stamps):
    # Reuse the loaded tables between screen updates; reread them when a CSV changes.
    return load_tables()


@st.cache_resource
def read_station_areas(stations_stamp):
    # Nearest-station areas inside the city boundary; rebuilt when sfd_stations.csv changes.
    stations = pd.read_csv(ROOT / "data/sfd_stations.csv")
    boundary = load_boundary()
    return build_station_areas(stations, boundary), boundary


@st.cache_resource
def read_prediction_geometries(stations_stamp):
    # Cache GeoJSON dicts, outer rings and interior label anchors once per stations file.
    areas, boundary = read_station_areas(stations_stamp)
    geoms, rings = area_geometries(areas)
    labels = area_labels(areas)
    return geoms, rings, labels, boundary


@st.cache_data
def estimate_for(stamps, model_stamps, day, station_ids):
    # One day's estimate per station, kept per date so nearby shift dates reuse it.
    features = read_data(stamps)[2]
    return learned_predictions(features, day, station_ids)[0].set_index("Station")["predicted_calls"]


@st.cache_data
def forecast_for(stamps, model_stamps, target, bands, station_ids):
    # Built once per date and data version (the stamps are only cache keys): a click on the
    # map only reruns the page.
    _, counts, features = read_data(stamps)
    usual = validate_predictions(
        reference_predictions(counts, target, station_ids), station_ids
    )
    # Always use the trained model for the supervisor-facing forecast.
    result, _ = learned_predictions(features, target, station_ids)
    # Match each station's estimate with its usual count to calculate the difference.
    view = result.merge(
        usual.rename(columns={"predicted_calls": "usual_calls"}),
        on="Station",
        validate="one_to_one",
    )
    view["difference"] = view["predicted_calls"] - view["usual_calls"]
    # Low and high: the estimate plus the usual misses for a forecast of that size.
    past = known_history(counts, target)
    # Last 4 weeks in the table: real calls, or this app's estimate for days after the data.
    dates, last = recent_days(target), counts["ShiftDate"].max()
    estimates = {
        d: estimate_for(stamps, model_stamps, d, station_ids) for d in dates if d > last
    }
    return (
        add_likely_range(view, bands),
        season_mix(past, target),
        same_weekday_calls(past, target),
        last_weeks(past, dates, last, estimates, station_ids),
    )


def zoom_safe_line_chart(container, series, tick_format="%b %d", **style):
    # A line chart you can scroll-zoom, but never zoom out past the data.
    container.plotly_chart(
        history_line_chart(series, tick_format, **style),
        width="stretch",
        config={"scrollZoom": True, "displayModeBar": False},
    )


def history_window(target, calls, recent, row, tone, scale):
    through = calls.index.max() if not calls.empty else None
    st.html(tb.block(*map(tb.lead, history_summary(target, recent, row, through)), card=True))
    span = st.segmented_control(
        "Time span",
        list(HISTORY_SPANS),
        default="4 weeks",
        required=True,
        key=SPAN_KEY,
        label_visibility="collapsed",
    )
    if calls.empty:
        st.html(
            tb.note("No finished shifts are known for this station before the planning time.")
        )
        return
    days = HISTORY_SPANS[span]
    st.html(history_legend_html(target, row["usual_calls"], tone, days is None or days > 91))
    st.plotly_chart(
        history_window_chart(
            calls,
            target,
            row["usual_calls"],
            row["predicted_calls"],
            row["low"],
            row["high"],
            days=days,
            tone=tone,
            scale=scale,
        ),
        width="stretch",
        config=PLOT_CONFIG,
    )
    st.html(
        tb.block(
            tb.note(
                "Only shifts finished by the planning time. Dashed stretches are shifts with "
                "missing data; a 0 means there really were no calls."
            ),
            tb.hints(["{point} Click outside the window or press {esc} to close."]),
        )
    )


def open_history(name, target, history, recent, row, scale):
    # Each window opens on 4 weeks, and its title names the station. The estimate takes the
    # busier or quieter colour of the station's card. The fill is rose at scale, the table's
    # Last 4 weeks top, so the window's colours match the station's row.
    st.session_state.pop(SPAN_KEY, None)
    calls = history.loc[history["Station"] == name].set_index("ShiftDate")["calls"]
    tone = range_bar_spec(
        row["predicted_calls"], row["low"], row["high"], row["usual_calls"]
    )["tone"]
    window = st.dialog(f"{name}: history known at planning time", width="large")
    window(history_window)(target, calls, recent, row, tone, scale)


def hourly_window(target, name, totals):
    try:
        # Use hourly_profile.csv shares to divide the shift estimate by hour. The citywide
        # split keeps its own colour scale; it is far bigger than any station.
        top = None
        if name == CITY:
            hourly = citywide_hourly(ROOT, target, totals)
        else:
            hourly = hourly_allocation(ROOT, target, name, totals[name])
            # The busiest hour of any station is rose, so SFD 16's quiet hours stay mint.
            top = scale_top(station_hour_peaks(ROOT, target, totals))
    except (OSError, ValueError, KeyError) as exc:
        st.info(str(exc))
        return
    st.altair_chart(hourly_altair_chart(hourly, top), width="stretch")
    st.html(
        tb.block(
            tb.lead(
                "Splits the shift total by each hour's usual share of calls, starting at 08:00."
            ),
            tb.hints(["{point} Click outside the window or press {esc} to close."]),
        )
    )


def open_hourly(name, target, totals):
    window = st.dialog(f"{name}: hourly breakdown, {target:%a %b %d}", width="large")
    window(hourly_window)(target, name, totals)


try:
    names = [
        "sfd_stations.csv",
        "station_shift_counts.csv",
        "modeling_dataset.csv",
        "forecast_features.csv",
    ]
    # File change times tell the cache when these CSVs need to be read again.
    stamps = tuple((ROOT / "data" / name).stat().st_mtime_ns for name in names)
    # The estimate also changes when the model files change.
    model_files = [
        ROOT / "predict.py",
        ROOT / "model_metadata.json",
        *sorted((ROOT / "models").glob("*")),
    ]
    model_stamps = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in model_files)
    # stations: sfd_stations.csv supplies station names (the area maps read it themselves).
    # counts: station_shift_counts.csv supplies past call totals and categories.
    # features: modeling_dataset.csv and forecast_features.csv supply the model's inputs.
    stations, counts, features = read_data(stamps)
    # Set the date picker's range: from where the model can serve to the last forecast date.
    min_date, max_date, default_date = target_dates(
        counts, first_model_date(), features["ShiftDate"].max().date()
    )
    # bands, city_range: prediction_ranges.csv supplies the likely range around each estimate.
    bands, city_range = load_ranges()
except (OSError, ValueError, KeyError) as exc:
    # Stop the page if required CSVs are missing or their contents do not agree.
    st.error(f"Cannot read the shared data: {exc}")
    st.stop()

# Station names from sfd_stations.csv, ordered by station number, for the compare list.
station_ids = sorted(stations["station"], key=lambda x: int(x.split()[-1]))
first = counts["ShiftDate"].min().date()
last = counts["ShiftDate"].max().date()

with st.container(key="top_bar"):
    title_col, date_col, meta_col = st.columns(
        [1.8, 1.0, 1.3], vertical_alignment="center"
    )
    with title_col:
        st.markdown("### CallCast Seattle")
        st.caption(
            "Expected calls per Seattle station area (08:00 to 08:00) · "
            f"Past 911 calls {first:%b %d, %Y} to {last:%b %d, %Y}"
        )
    with date_col:
        target_date = st.date_input(
            "Shift date",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
        )
    # For a Saturday target, show Thursday at 08:00 as the time the supervisor would plan.
    origin = forecast_origin(target_date)
    with meta_col:
        st.markdown(f"**Planned {origin:%a %b %d, %H:%M}**")
        st.caption(f"Estimates are available up to {max_date:%b %d, %Y}.")

prediction_tab, explore_tab, about_tab = st.tabs(
    ["Prediction", "Explore", "About"]
)
# Keep only completed earlier shifts from the counts CSV, excluding suspected outages.
history = known_history(counts, target_date)

with prediction_tab:
    end = pd.Timestamp(target_date) + pd.Timedelta(days=1)
    # After the last day of counts the shift has not happened yet, so it is labelled Forecast.
    is_forecast = target_date > last
    # Past the first RECENT_CALL_SHIFTS forecast shifts, no recent calls are known, so the
    # model's recent-call inputs are typical values for the station, month and weekday.
    is_long_range = target_date > last + pd.Timedelta(days=RECENT_CALL_SHIFTS)
    shift_heading = (
        f"{target_date:%a %b %d} 08:00 → {end:%a %b %d} 08:00"
        + ("  :red[Forecast]" if is_forecast else "")
        + (" · long-range estimate" if is_long_range else "")
    )
    picked, refused, map_layer_id = sync_map_picks(
        st.session_state, MAP_KEY, PICK_KEY, CLICKED_KEY, PICK_GEN_KEY
    )
    # A tick in the ranked table picks or unpicks that row's station, the same as the map. A
    # click on its Last 4 weeks or Expected names the window to open below the table.
    table_click = (st.session_state.get(TABLE_KEY) or {}).get("action")
    picked, table_refused, table_window = table_action(table_click, picked, station_ids)
    st.session_state[PICK_KEY] = picked
    view = None
    error_text = None
    try:
        # Built once per date: estimates, usual calls, the call type mix, each station's
        # last 4 shifts on this weekday and every day of the 4 weeks before planning.
        view, mix, recent, weeks = forecast_for(
            stamps, model_stamps, target_date, bands, tuple(station_ids)
        )
    except (
        OSError,
        ValueError,
        KeyError,
        AttributeError,
        ImportError,
        TypeError,
    ) as exc:
        error_text = str(exc)

    shift_tone = "usual"
    if view is not None:
        city_diff = float(view["difference"].sum())
        if city_diff > BRIEF_USUAL_CALLS:
            shift_tone = "busier"
        elif city_diff < -BRIEF_USUAL_CALLS:
            shift_tone = "quieter"

    def _draw_shift_header(container):
        container.subheader(shift_heading)
        with container.container(key="prediction_briefing"):
            st.html(
                briefing_html(
                    shift_briefing(target_date, view, error_text), tone=shift_tone
                )
            )
        if error_text and error_text != GAP_MESSAGE:
            container.warning(error_text)
        # Weather alerts are citywide, so they sit under the briefing; like event warnings,
        # they are shown next to the estimate and never change it.
        now = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None)
        # The shift's weather icon, low and high, from the same forecast as the alerts; no line
        # when there is no forecast for the shift.
        weather_line = weather_now_html(get_shift_weather(target_date, now))
        if weather_line:
            with container.container(key="prediction_weather_now"):
                st.html(weather_line)
        weather = get_weather_alerts(target_date, now)
        with container.container(key="prediction_weather"):
            if weather["status"] == "alert":
                # Each alert gets its icon at the left (a snowflake for a hard freeze); none when
                # the icon file is missing.
                icon_css = alert_icon_css(weather["kinds"])
                if icon_css:
                    st.html(icon_css)
                for kind, alert in zip(weather["kinds"], weather["messages"]):
                    with st.container(key=f"weather_alert_{kind}"):
                        st.warning(alert)
                st.html(tb.note(WEATHER_NOTE, under=True))
            else:
                st.html(tb.note(weather["messages"][0]))
            # Snow and ice are not in our alerts, so point to the official NWS ones for upcoming shifts.
            if show_nws(target_date, now):
                # The NWS logo in front of the note; the plain note when the logo file is missing.
                nws, nws_css = nws_note_html(tb.note(NWS_NOTE))
                if nws_css:
                    st.html(nws_css)
                with st.container(key="prediction_nws"):
                    st.html(nws)
        # What the numbers are based on: the data date as a small fact, and for a forecast the
        # reasons behind a "why" toggle.
        why = []
        if is_forecast:
            why.append(
                f"**Forecast:** this shift is after the last day of data ({last:%b %d, %Y}), "
                "so there are no actual calls to compare it with."
            )
        if is_long_range:
            why.append(
                "**Long-range estimate:** no calls close to this date are known yet, so the "
                "estimate uses the typical pattern for this station, month, weekday and holidays. "
                "It is less precise than an estimate for the next few days."
            )
        parts = [
            tb.facts(
                [
                    (
                        "Data through",
                        f"**{data_through(history)}**, the last shift finished by the planning "
                        "time.",
                    )
                ]
            )
            if not history.empty
            else tb.note("No completed shifts before the planning time yet.")
        ]
        if why:
            title = "Why a long-range estimate" if is_long_range else "Why a forecast"
            parts.append(tb.more(why, title))
        container.html(tb.block(*parts))
        return weather["status"] == "alert"

    if view is not None:
        by_station = view.set_index("Station")
        total = float(view["predicted_calls"].sum())
        usual_total = float(view["usual_calls"].sum())
        city_low, city_high = citywide_range(total, city_range)
        move_label, mover = biggest_move(view)

        # Top: the map on the left, the shift summary and cards on the right.
        with st.container(key="prediction_split"):
            col_map, col_info = st.columns([0.95, 1.45], gap="large")
            with col_map, st.container(key="prediction_map_panel"):
                st.markdown("##### Expected calls compared with usual, by station area")
                if refused:
                    st.warning(refusal_text(refused))
                geoms, rings, labels, boundary = read_prediction_geometries(stamps[0])
                hover = {
                    name: {
                        "expected": f"{r['predicted_calls']:.1f}",
                        "range": f"{int(r['low'])} to {int(r['high'])}",
                        "usual": f"{r['usual_calls']:.1f}",
                        "diff": f"{r['difference']:+.1f}",
                        "status": station_status_words(r["difference"]),
                    }
                    for name, r in by_station.iterrows()
                }
                outlines = {
                    name: station_color(picked, name, alpha=255) for name in picked
                }
                # The key keeps the map mounted when the date, and with it the shading, changes.
                # The height is the least that shows the whole city; on wide screens style.css
                # stretches the map so the left column ends level with the right one.
                with st.container(key="prediction_map_box"):
                    st.pydeck_chart(
                        forecast_map(
                            geoms,
                            rings,
                            boundary,
                            fills=difference_fills(view),
                            labels=labels,
                            expected=by_station["predicted_calls"].to_dict(),
                            picked_colors=outlines,
                            details=hover,
                            layer_id=map_layer_id,
                        ),
                        height=fixed_view(boundary)[1],
                        on_select="rerun",
                        selection_mode="multi-object",
                        key=MAP_KEY,
                    )
                st.html(map_legend_html())
                st.caption("Point for details; click up to 3 to compare.")
                st.multiselect(
                    f"Compare stations (up to {MAX_COMPARE})",
                    station_ids,
                    key=PICK_KEY,
                    max_selections=MAX_COMPARE,
                    placeholder="Choose stations, or click areas on the map",
                )

            with col_info, st.container(key="prediction_top_info"):
                weather_alert = _draw_shift_header(st)
                # Three top cards: citywide, biggest shift vs usual, and stations to watch.
                col_city, col_busiest, col_watch = st.columns(3)
                with (
                    col_city,
                    st.container(border=True, height="stretch", key="card_citywide"),
                ):
                    st.metric(
                        "Citywide expected calls",
                        f"{total:.1f}",
                        delta=f"{view['difference'].sum():+.1f} vs usual",
                        delta_color="off",
                    )
                    st.html(
                        range_bar_html(
                            range_bar_spec(
                                total,
                                city_low,
                                city_high,
                                usual_total,
                                threshold=BRIEF_USUAL_CALLS,
                            )
                        )
                    )
                with (
                    col_busiest,
                    st.container(border=True, height="stretch", key="card_busiest"),
                ):
                    st.metric(
                        f"{move_label}: {mover['Station']}",
                        f"{mover['predicted_calls']:.1f}",
                        delta=f"{mover['difference']:+.1f} vs usual",
                        delta_color="off",
                    )
                    st.html(
                        range_bar_html(
                            range_bar_spec(
                                mover["predicted_calls"],
                                mover["low"],
                                mover["high"],
                                mover["usual_calls"],
                            )
                        )
                    )
                with (
                    col_watch,
                    st.container(border=True, height="stretch", key="card_watch"),
                ):
                    st.markdown("##### Stations to watch")
                    watch = stations_to_watch(view)
                    if watch.empty:
                        st.html(
                            tb.note("No station is clearly busier than usual for this shift.")
                        )
                    for _, row in watch.iterrows():
                        st.markdown(
                            f"- **{row['Station']}**: {row['predicted_calls']:.1f} vs {row['usual_calls']:.1f} usual"
                        )
                    if not watch.empty:
                        st.html(
                            tb.more(
                                f"Shown when the estimate is at least {WATCH_MIN_EXTRA} calls "
                                f"and {WATCH_MIN_SHARE:.0%} above the usual for this weekday. "
                                f"At most {WATCH_MAX} are listed.",
                                "How they are picked",
                            )
                        )

                any_warning = False
                if picked:
                    st.subheader("Comparing " + ", ".join(picked))
                    # Each card's top edge has its station colour, and a round badge in the
                    # corner shows its pick number, the same as its tag and table row.
                    rim_css = "".join(
                        f".st-key-card_compare_{i} {{ border-top: 4px solid {station_color(picked, name)} !important; position: relative; }}"
                        f'.st-key-card_compare_{i}::before {{ content: "{i + 1}"; {PICK_BADGE} '
                        f"background: var(--cc-pick-{i}); color: var(--cc-pick-text-{i}); }}"
                        for i, name in enumerate(picked)
                    )
                    st.html(f"<style>{rim_css}</style>")
                    for i, (column, name) in enumerate(
                        zip(st.columns(len(picked)), picked)
                    ):
                        row = by_station.loc[name]
                        with (
                            column,
                            st.container(
                                border=True, height="stretch", key=f"card_compare_{i}"
                            ),
                        ):
                            st.metric(
                                f"{name} expected calls",
                                f"{row['predicted_calls']:.1f}",
                                delta=f"{row['difference']:+.1f} vs usual",
                                delta_color="off",
                            )
                            st.html(
                                range_bar_html(
                                    range_bar_spec(
                                        row["predicted_calls"],
                                        row["low"],
                                        row["high"],
                                        row["usual_calls"],
                                    )
                                )
                            )
                            # Event warnings are informational and do not change the estimate.
                            for warning in get_event_warnings(target_date, name):
                                st.warning(warning)
                                any_warning = True
                    if any_warning:
                        st.html(
                            tb.note(
                                "Event warnings are informational and are not included in the "
                                "forecast."
                            )
                        )

        # Below the map, full width: call type mix, then the ranked table and its download.
        with st.container(key="prediction_bottom_info"):
            names = picked or [CITY]
            mix_df = mix_table(mix, names)
            with st.container(border=True, key="card_mix"):
                st.markdown("##### Call type mix (average calls per shift)")
                mix_chart_col, mix_table_col = st.columns(
                    [1.6, 1.0], vertical_alignment="center"
                )
                mix_chart_col.plotly_chart(
                    call_mix_chart(
                        mix_df,
                        [station_color(picked, n) for n in picked] if picked else None,
                    ),
                    width="stretch",
                    config=PLOT_CONFIG,
                )
                mix_table_col.dataframe(
                    mix_df,
                    width="stretch",
                    column_config={
                        n: st.column_config.NumberColumn(format="%.1f") for n in names
                    },
                )
                st.html(tb.note("From past shifts in this season, not a forecast."))

            st.markdown("##### All stations, busiest first")
            # Name only what the alert and event notes above have not already said is left out
            # (Oct 25 with SFD 10 picked: "The estimate does not include weather.").
            said = {"weather": weather_alert, "events such as games": any_warning}
            left_out = [part for part, done in said.items() if not done]
            estimate_note = (
                [tb.note(f"The estimate does not include {' or '.join(left_out)}.")]
                if left_out
                else []
            )
            # One card: the column words, a note and links, then how to use the table's controls.
            st.html(
                tb.block(
                    tb.terms(
                        [
                            (
                                "Likely range",
                                "about 4 in 5 similar station shifts and about 7 in 10 citywide "
                                "totals ended up inside it. It shows how much the estimate can "
                                "vary and is not a staffing recommendation.",
                            ),
                            ("Usual", "The average of earlier shifts on this weekday."),
                            ("vs usual", "Expected minus usual."),
                        ]
                    ),
                    tb.row(*estimate_note, tb.chips(LINKS)),
                    tb.hints(
                        [
                            "{tick} Tick a row to compare that station, the same as clicking its "
                            "area on the map.",
                            "{bars} Click a station's Last 4 weeks bars for its history, or its "
                            "{track} Expected bar for the hourly breakdown.",
                            "{download} The download keeps the likely range as Low and High.",
                        ]
                    ),
                    card=True,
                )
            )
            if table_refused:
                st.warning(refusal_text(table_refused))
            table_html = ranked_table_html(
                ranked_rows(view, weeks), picked, list(weeks.columns), last, target_date
            )
            ranked_table(
                key=TABLE_KEY,
                data={"html": table_html, "limit": MAX_COMPARE},
                on_action_change=lambda: None,
            )
            with st.container(horizontal=True):
                st.download_button(
                    "Download displayed counts",
                    download_rows(view, target_date, origin).to_csv(index=False),
                    f"callcast_counts_{target_date}.csv",
                    "text/csv",
                    type="primary",
                )
                city_hours = st.button("Citywide hourly breakdown", key="city_hourly")
            totals = by_station["predicted_calls"]
            kind, name = table_window or (None, None)
            if kind == "history":
                row = by_station.loc[name]
                open_history(
                    name, target_date, history, recent.get(name, []), row,
                    scale_top(weeks.to_numpy().ravel()),
                )
            elif kind == "hourly":
                open_hourly(name, target_date, totals)
            elif city_hours:
                open_hourly(CITY, target_date, totals)
    else:
        # No estimate (Jan 9, 2026 has a gap in the call data): the same split and map, every
        # area in one pale shade with its number only; the briefing says why.
        with st.container(key="prediction_split"):
            col_map, col_info = st.columns([0.95, 1.45], gap="large")
            with col_map, st.container(key="prediction_map_panel"):
                st.markdown("##### Station areas")
                if refused:
                    st.html(tb.note(refusal_text(refused)))
                geoms, rings, labels, boundary = read_prediction_geometries(stamps[0])
                outlines = {
                    name: station_color(picked, name, alpha=255) for name in picked
                }
                with st.container(key="prediction_map_box"):
                    st.pydeck_chart(
                        forecast_map(
                            geoms,
                            rings,
                            boundary,
                            fills=None,
                            labels=labels,
                            picked_colors=outlines,
                            layer_id=map_layer_id,
                        ),
                        height=fixed_view(boundary)[1],
                        on_select="rerun",
                        selection_mode="multi-object",
                        key=MAP_KEY,
                    )
                st.caption("Point for a station; click up to 3 to keep them for another date.")
                st.multiselect(
                    f"Compare stations (up to {MAX_COMPARE})",
                    station_ids,
                    key=PICK_KEY,
                    max_selections=MAX_COMPARE,
                    placeholder="Choose stations, or click areas on the map",
                )
            with col_info, st.container(key="prediction_top_info"):
                _draw_shift_header(st)
                # The briefing already says why there is no estimate; this only adds where to look.
                st.html(
                    tb.block(
                        tb.row(
                            tb.note("Past calls for each station are on the Explore tab."),
                            tb.chips(LINKS),
                        )
                    )
                )


with explore_tab:
    # Build comparisons from the same earlier, unmasked counts used in the Prediction tab.
    st.subheader("Patterns before the selected planning time")
    # Say which shifts the charts use: they stop at the planning time, and dates past the
    # data all use the same, latest history.
    if not history.empty:
        through = history["ShiftDate"].max()
        note = f"Citywide and Stations use finished shifts up to {through:%a %b %d, %Y}"
        # Compare with the last day of counts, not `through`: an outage shift just before the
        # planning time also makes `through` earlier, but newer data still exists.
        if target_date - pd.Timedelta(days=3) > last:
            # Past the data: easy to miss, so show it as a notice in this tab only (the
            # Prediction tab uses future dates on purpose, so no pop-up there).
            st.info(
                f"The selected date is after the last day of data ({through:%b %d, %Y}). "
                f"{note}, so every later date shows the same history until new calls are "
                "added. The estimate for this date is on the Prediction tab.",
                icon=":material/info:",
            )
        else:
            st.html(
                tb.note(
                    note
                    + " (the last one finished by the planning time), without shifts that have "
                    "missing data."
                )
            )
    # Four views: the whole city (default), station areas and comparisons, past events,
    # and past weather.
    city_view, stations_view, events_view, weather_view = st.tabs(
        ["Citywide", "Stations", "Events", "Weather"]
    )
    with city_view:
        if history.empty:
            st.html(tb.note("No history is available for this selection."))
        else:
            # One choice drives the map and the charts below: all calls or one call type.
            choice = (
                st.segmented_control(
                    "Show calls for",
                    list(CALL_TYPES),
                    default="All calls",
                    key="city_call_type",
                )
                or "All calls"
            )
            column, words = CALL_TYPES[choice]
            one = [CITY]
            city = only_type(citywide(history), column)
            # Map: each station area shaded by its average of the chosen calls per shift.
            per_station = history.groupby("Station")[column].mean().reindex(station_ids)
            areas, boundary = read_station_areas(stamps[0])
            with st.container(key="explore_card_city_map"):
                st.markdown(f"##### Average {words} per shift by station area")
                # A click opens the area's history window (all calls, as from the table's bars).
                # The layer id then moves on, so each click is read once and never reopens it.
                gen = st.session_state.get(CITY_GEN_KEY, 0)
                layer_id = f"{AREA_LAYER_ID}-{gen}" if gen else AREA_LAYER_ID
                _, clicked = area_selection(st.session_state.get(CITY_MAP_KEY), layer_id)
                if clicked:
                    gen += 1
                    st.session_state[CITY_GEN_KEY] = gen
                st.pydeck_chart(
                    value_map(
                        areas,
                        boundary,
                        per_station,
                        {s: f"{v:.1f} {words} per shift" for s, v in per_station.items()},
                        layer_id=f"{AREA_LAYER_ID}-{gen}" if gen else AREA_LAYER_ID,
                    ),
                    on_select="rerun",
                    selection_mode="single-object",
                    key=CITY_MAP_KEY,
                )
                if clicked and view is not None:
                    open_history(
                        clicked[0],
                        target_date,
                        history,
                        recent.get(clicked[0], []),
                        by_station.loc[clicked[0]],
                        scale_top(weeks.to_numpy().ravel()),
                    )
                top = per_station.nlargest(3)
                st.html(
                    tb.block(
                        tb.lead(
                            f"Lighter areas have fewer {words}, rose areas more: from "
                            f"{per_station.min():.1f} ({per_station.idxmin()}) to "
                            f"{per_station.max():.1f} per shift."
                        ),
                        tb.terms([("Most", ", ".join(f"{s} ({v:.1f})" for s, v in top.items()))]),
                        tb.hints(["{point} Point at an area for its number."]),
                    )
                )
            st.markdown(f"##### Citywide {words} per shift")
            st.html(tb.lead("All 34 station areas added together, per shift (08:00 to 08:00)."))
            st.dataframe(compare_summary(city, one, average=False), width="stretch")
            with st.container(key="explore_card_city_trend"):
                st.markdown(
                    f"##### Citywide {words} per shift, last {TREND_DAYS} days (7-shift average)"
                )
                if choice == "All calls":
                    # All calls: stacked bands by type, so a rise shows which type caused it.
                    st.html(
                        tb.block(
                            tb.lead(
                                "Each band is one call type; the top edge is all calls. Each point "
                                "averages the 7 shifts up to that day."
                            ),
                            tb.hints(["{point} Point at the chart to see every type's number."]),
                        )
                    )
                    figure = stacked_trend_chart(type_trend(citywide(history)))
                else:
                    st.html(
                        tb.block(
                            tb.lead(
                                "Each point averages the 7 shifts up to that day, which smooths out "
                                "day-to-day swings."
                            ),
                            tb.note("Look for lasting rises or drops, not single bumps."),
                        )
                    )
                    figure = trend_chart(trend(city, one, average=False))
                st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG)
            a, b = st.columns(2)
            with a.container(key="explore_card_city_weekday", height="stretch"):
                st.markdown(f"##### Average {words} per shift by weekday")
                st.html(tb.lead("Taller bars are busier days of the week."))
                st.plotly_chart(
                    weekday_chart(weekday_means(city, one, average=False)),
                    width="stretch",
                    config=PLOT_CONFIG,
                )
            with b.container(key="explore_card_city_month", height="stretch"):
                st.markdown(f"##### Average {words} per shift by month")
                st.html(tb.lead("Each point is one month since 2022: shows seasons and longer-term changes."))
                # Averaged per shift, so longer months do not win by size alone.
                zoom_safe_line_chart(
                    st, monthly(city), "%b %Y", color=CITY_COLOR, show_markers=True
                )
            with st.container(key="explore_card_city_types"):
                st.markdown("##### Share of calls by category")
                st.html(
                    tb.block(
                        tb.lead(
                            "How the city's calls split by type (all types, whatever is chosen above)."
                        ),
                        tb.hints(["{point} Point at a bar to see the average number of calls per shift."]),
                    )
                )
                all_types = citywide(history)
                st.plotly_chart(
                    category_chart(
                        category_shares(all_types, one, average=False),
                        category_means(all_types, one, average=False),
                    ),
                    width="stretch",
                    config=PLOT_CONFIG,
                )
            st.html(tb.note("The Stations tab shows each station area and compares stations."))
    with stations_view:
        # Station area map: city boundary and nearest-station areas, with recent averages on hover.
        with st.container(key="explore_card_station_map"):
            st.markdown("##### Station areas")
            picked, refused = [], []
            if not history.empty:
                areas, boundary = read_station_areas(stamps[0])
                # Shade each area by the share of its calls of one type (shares, not counts, so
                # small and busy stations are on the same scale).
                share_choice = (
                    st.segmented_control(
                        "Color areas by",
                        list(SHARE_TYPES),
                        default="Fire share",
                        key="station_share_type",
                    )
                    or "Fire share"
                )
                share_column = SHARE_TYPES[share_choice]
                shares, city_share = station_shares(history, share_column)
                share_word = share_choice.split()[0]
                hover = {
                    s: f"{share_word} share: {v:.0f}% (city {city_share:.0f}%)"
                    for s, v in shares.items()
                }
                picked, refused, explore_layer_id = sync_map_picks(
                    st.session_state,
                    EXPLORE_MAP_KEY,
                    EXPLORE_PICK_KEY,
                    EXPLORE_CLICKED_KEY,
                    EXPLORE_GEN_KEY,
                )
                outlines = {name: station_color(picked, name, alpha=255) for name in picked}
                # Click areas to pick stations; clicking a picked area again unpicks it.
                st.pydeck_chart(
                    area_map(
                        areas,
                        boundary,
                        stations,
                        recent_averages(history),
                        fills=shade(shares),
                        picked_colors=outlines,
                        extra=hover,
                        layer_id=explore_layer_id,
                    ),
                    on_select="rerun",
                    selection_mode="multi-object",
                    key=EXPLORE_MAP_KEY,
                )
                high = shares.nlargest(2)
                st.html(
                    tb.block(
                        tb.lead(
                            f"Rose areas have a larger share of {share_word.lower()} calls: from "
                            f"{shares.min():.0f}% to {shares.max():.0f}% (city {city_share:.0f}%)."
                        ),
                        tb.terms(
                            [("Highest", ", ".join(f"{s} ({v:.0f}%)" for s, v in high.items()))]
                        ),
                        tb.hints(
                            [
                                "{point} Point at an area for its share and its average calls over "
                                "the last week and month.",
                                f"{{area}} Click up to {MAX_COMPARE} areas to compare them (outlined "
                                "in color); click an area again to remove it.",
                            ]
                        ),
                        tb.note(
                            "Each area is the part of Seattle closest to that station, so its edges "
                            "are approximate."
                        ),
                    )
                )
            else:
                st.session_state[EXPLORE_CLICKED_KEY] = []
                st.session_state[EXPLORE_GEN_KEY] = (0, None)
        if refused:
            verb = "was" if len(refused) == 1 else "were"
            st.warning(
                f"Up to {MAX_COMPARE} stations can be compared at once, so {', '.join(refused)} "
                f"{verb} not added. Click a picked area again to remove it."
            )
        if picked:
            # 1 to MAX_COMPARE picked stations, each chart with the average station as a reference.
            st.subheader(
                picked[0] if len(picked) == 1 else "Comparing " + ", ".join(picked)
            )
            st.html(
                tb.terms(
                    [
                        (
                            "{dash} Average station",
                            f"Citywide calls divided by {len(station_ids)} stations, so it is on "
                            "the same scale as one station.",
                        )
                    ]
                )
            )
            st.dataframe(compare_summary(history, picked), width="stretch")
            with st.container(key="explore_card_pick_trend"):
                st.markdown(
                    f"##### Calls per shift, last {TREND_DAYS} days (7-shift average)"
                )
                st.html(
                    tb.block(
                        tb.lead(
                            "One line per station; the grey dashed line is the average station. Each "
                            "point averages the 7 shifts up to that day."
                        ),
                        tb.note("Look for lasting rises or drops."),
                    )
                )
                st.plotly_chart(
                    trend_chart(trend(history, picked)), width="stretch", config=PLOT_CONFIG
                )
            with st.container(key="explore_card_pick_month"):
                st.markdown("##### Average calls per shift by month (all history)")
                st.html(
                    tb.lead(
                        "Each point is one month since 2022: shows seasons and whether a station "
                        "has become busier or quieter over the years."
                    )
                )
                st.plotly_chart(
                    monthly_chart(monthly_means(history, picked)),
                    width="stretch",
                    config=PLOT_CONFIG,
                )
            with st.container(key="explore_card_pick_weekday"):
                st.markdown("##### Average calls per shift by weekday")
                st.html(
                    tb.lead("Taller bars are busier days; compare each station with the striped average.")
                )
                st.plotly_chart(
                    weekday_chart(weekday_means(history, picked)),
                    width="stretch",
                    config=PLOT_CONFIG,
                )
            with st.container(key="explore_card_pick_types"):
                st.markdown("##### Share of calls by category")
                st.html(
                    tb.block(
                        tb.lead("Shares, so busy and quiet stations can be compared directly."),
                        tb.hints(
                            ["{point} Point at a bar to see the average number of calls per shift too."]
                        ),
                    )
                )
                st.plotly_chart(
                    category_chart(
                        category_shares(history, picked), category_means(history, picked)
                    ),
                    width="stretch",
                    config=PLOT_CONFIG,
                )
        elif not history.empty:
            st.html(
                tb.block(
                    tb.hints(
                        [
                            f"{{area}} Click up to {MAX_COMPARE} station areas on the map to see "
                            "and compare their history. The Citywide tab shows the whole city."
                        ]
                    )
                )
            )
        if not history.empty:
            # Every station at a glance, in number order; picked stations keep their colours.
            with st.container(key="explore_card_station_bars"):
                st.markdown("##### Average calls per shift by station")
                st.html(
                    tb.lead(
                        "One bar per station, in number order. Bars above the dashed line are busier "
                        "than the average station; picked stations are shown in their colors."
                    )
                )
                st.plotly_chart(
                    station_bar_chart(station_means(history, station_ids), picked),
                    width="stretch",
                    config=PLOT_CONFIG,
                )
    with events_view:
        # Past event effects: a venue map over the same station areas, then the tables.
        show_event_effects(st, *read_station_areas(stamps[0]))
    with weather_view:
        # Past weather effects for the whole city, from data/weather_*.csv.
        show_weather_effects(st)

with about_tab:
    # Text only, as cards: sources and links, what the numbers mean, the likely range, dates far
    # ahead, what is left out, and what the numbers don't show.
    dataset = "https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj"
    st.html(
        tb.panels(
            tb.block(
                tb.terms(
                    [
                        ("911 calls", "Seattle Real Time Fire 911 Calls"),
                        ("Station locations", "Seattle GIS"),
                    ]
                ),
                tb.chips([("Source", "911 dataset", dataset), *LINKS]),
                tb.note("This is a fixed set of past calls, not live 911 data."),
                card=True,
                title="Sources",
            ),
            tb.block(
                tb.terms(
                    [
                        (
                            "Each number",
                            "The expected count of 911 calls for one station area in one shift, "
                            "08:00 to 08:00 the next day.",
                        ),
                        (
                            "Station area",
                            "The area closest to that fire station, so its edges are approximate.",
                        ),
                        (
                            "Usual for this weekday",
                            "The average of earlier shifts on the same weekday.",
                        ),
                        (
                            "Planning time",
                            "Every estimate uses only shifts that had finished by the planning "
                            "time.",
                        ),
                    ]
                ),
                card=True,
                title="What the numbers mean",
            ),
            tb.block(
                tb.lead(
                    "The likely range comes from how far our estimates were from what really "
                    "happened across 2025, for stations with a similar expected count. About 4 in "
                    "5 of those station shifts ended up inside it, and about 7 in 10 for the "
                    "citywide total."
                ),
                tb.note(
                    "Some shifts will still land outside it, and it does not say how many units "
                    "to send."
                ),
                card=True,
                title="About the likely range",
            ),
            tb.block(
                tb.lead("Estimates go up to 90 days after the last data."),
                tb.terms(
                    [
                        (
                            "More than 3 days after",
                            "The most recent calls are not known. The estimate then uses the "
                            "typical values for that station, month and weekday, plus holidays, "
                            "and is marked as a long-range estimate.",
                        )
                    ]
                ),
                tb.note(
                    "The model itself is not retrained, so it does not learn from calls after "
                    "2025."
                ),
                card=True,
                title="Dates far ahead",
            ),
            tb.block(
                tb.terms(
                    [
                        ("Practice calls", "TEST"),
                        ("Duplicate follow-up records", "Fast Back Up"),
                        (
                            "Two referral types",
                            "Triaged Incident and Nurseline/AMR, which stopped being recorded "
                            "after May 2025",
                        ),
                        ("Data gaps", "Shifts where the data feed had a gap"),
                    ]
                ),
                card=True,
                title="What is left out",
            ),
            tb.block(
                tb.lead("How long a call takes, which units are free, or travel time."),
                tb.note("Staffing is the supervisor's decision."),
                card=True,
                title="What the numbers don't show",
            ),
        )
    )
