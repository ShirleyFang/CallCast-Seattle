"""Build the CallCast Seattle page. Run with: uv run streamlit run app.py"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_data import (
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
    target_dates,
    validate_predictions,
)
from station_areas import (
    area_map,
    build_station_areas,
    load_boundary,
    recent_averages,
    selected_stations,
    shade,
    value_map,
)
from event_explore import show_event_effects
from event_warnings import get_event_warnings
from station_forecast import (
    DIVERGE_STOPS,
    WATCH_MAX,
    WATCH_MIN_EXTRA,
    WATCH_MIN_SHARE,
    data_through,
    difference_fills,
    mix_table,
    season_mix,
    stations_to_watch,
    toggle_picks,
)
from station_compare import (
    CALL_TYPES,
    CITY,
    SHARE_TYPES,
    CITY_COLOR,
    MAX_COMPARE,
    PLOT_CONFIG,
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

# Plain-word column names for the table and the download.
COLUMN_NAMES = {
    "predicted_calls": "Expected calls",
    "usual_calls": "Usual for this weekday",
    "difference": "Difference",
    "low": "Low",
    "high": "High",
    "ShiftDate": "Shift date",
    "AsOf": "Planning time",
}
# Column order for the table and the download.
TABLE_COLUMNS = [
    "Station",
    "predicted_calls",
    "low",
    "high",
    "usual_calls",
    "difference",
]

# Session keys: the picked stations (the multiselect) and the Prediction map.
PICK_KEY = "compare_stations"
MAP_KEY = "prediction_map"
# The map's clicked areas at the last run, to tell which area a new click changed.
CLICKED_KEY = "map_clicked"
# Light gold behind the picked stations' rows in the table.
PICK_HIGHLIGHT = "background-color: #F3E3B8"

# The first 3 shifts after the last day of counts use real recent calls (lag3 is known);
# later shifts use typical values instead (MIN_LAG and data/lag_fill_means.csv in step 4).
RECENT_CALL_SHIFTS = 3

# Outside pages a planner can open for live incidents and for station coverage.
LINKS = (
    "Live incidents: [SFD Live](https://sfdlive.com/) · "
    "Station coverage: [Seattle Breaking guide]"
    "(https://www.seattlebreaking.com/articles/seattle-fire-stations-coverage)"
)

st.set_page_config(page_title="CallCast Seattle", layout="wide")
# Hover colors and the hand cursor for clickable parts live in style.css.
st.html(f"<style>{(ROOT / 'style.css').read_text()}</style>")


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


@st.cache_data
def forecast_for(stamps, model_stamps, target, bands, station_ids):
    # Built once per date and data version (the stamps are only cache keys): a click on the
    # map only reruns the page.
    _, counts, features = read_data(stamps)
    usual = validate_predictions(reference_predictions(counts, target, station_ids), station_ids)
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
    return add_likely_range(view, bands), season_mix(known_history(counts, target), target)


def zoom_safe_line_chart(container, series, tick_format="%b %d", color="#2F5D8A"):
    """Draw a line chart you can scroll-zoom, but never zoom out past the data."""
    # minallowed and maxallowed stop the zoom at the data. The default chart let the line
    # shrink to a sliver, with a year of empty dates and negative call counts around it.
    first, last, top = series.index.min(), series.index.max(), float(series.max()) * 1.1
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            mode="lines",
            line={"color": color, "width": 2},
            connectgaps=False,
            hovertemplate="%{x|" + tick_format + "}: %{y:.1f} calls<extra></extra>",
        )
    )
    fig.update_layout(
        height=350,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1F2A36"},
        dragmode="pan",
        xaxis={
            "tickformat": tick_format,
            "range": [first, last],
            "minallowed": first,
            "maxallowed": last,
            "gridcolor": "#E6E1D9",
        },
        yaxis={
            "range": [0, top],
            "minallowed": 0,
            "maxallowed": top,
            "gridcolor": "#E6E1D9",
        },
    )
    container.plotly_chart(
        fig, width="stretch", config={"scrollZoom": True, "displayModeBar": False}
    )


st.title("CallCast Seattle")
st.write(
    "Expected calls per Seattle station area for one shift, 08:00 to 08:00 the next day."
)
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
    model_files = [ROOT / "predict.py", ROOT / "model_metadata.json", *sorted((ROOT / "models").glob("*"))]
    model_stamps = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in model_files)
    # stations: sfd_stations.csv supplies station names and map coordinates.
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
st.caption(
    f"Past 911 calls from {first:%b %d, %Y} to {last:%b %d, %Y} • {len(station_ids)} station areas"
)
target_date = st.sidebar.date_input(
    "Target shift date",
    value=default_date,
    min_value=min_date,
    max_value=max_date,
)
st.sidebar.caption(f"Estimates are available up to {max_date:%b %d, %Y}.")

# For a Saturday target, show Thursday at 08:00 as the time the supervisor would plan.
origin = forecast_origin(target_date)
st.sidebar.write(f"Planning time: {origin:%a %b %d, %Y at %H:%M %Z}")
st.sidebar.caption(
    "Only uses shifts that had finished by the planning time (Thursday 08:00 for a Saturday shift)."
)

prediction_tab, explore_tab, about_tab = st.tabs(
    ["Prediction", "Explore", "Data and limits"]
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
    st.subheader(
        f"{target_date:%a %b %d} 08:00 → {end:%a %b %d} 08:00"
        + ("  :red[Forecast]" if is_forecast else "")
        + (" · long-range estimate" if is_long_range else "")
    )
    if is_forecast:
        st.caption(
            f"Forecast: this shift is after the last day of data ({last:%b %d, %Y}), "
            "so there are no actual calls to compare it with."
        )
    if is_long_range:
        st.caption(
            "Long-range estimate: no calls close to this date are known yet, so the estimate "
            "uses the typical pattern for this station, month, weekday and holidays. "
            "It is less precise than an estimate for the next few days."
        )
    st.caption(
        f"Data through {data_through(history)}, the last shift finished by the planning time."
        if not history.empty
        else "No completed shifts before the planning time yet."
    )
    # The map keeps its own clicked areas, but the multiselect is the only place the picked
    # stations are kept: an area whose click state changed since the last run was clicked
    # once, and that click picks or unpicks its station.
    on_map = selected_stations(st.session_state.get(MAP_KEY))
    # Sorted so two clicks that land in one run always give the same result.
    clicked = sorted(set(on_map) ^ set(st.session_state.get(CLICKED_KEY, [])))
    st.session_state[CLICKED_KEY] = on_map
    refused = []
    picks = st.session_state.get(PICK_KEY, [])
    if clicked:
        picks, refused = toggle_picks(picks, clicked)
    # Saved on every run: on a date with no estimate the list is not drawn, and Streamlit
    # would otherwise drop the picks with it.
    st.session_state[PICK_KEY] = picks
    picked = picks
    view = None
    try:
        # Estimates, usual calls and the call type mix, built once per date.
        view, mix = forecast_for(stamps, model_stamps, target_date, bands, tuple(station_ids))
    except (
        OSError,
        ValueError,
        KeyError,
        AttributeError,
        ImportError,
        TypeError,
    ) as exc:
        st.warning(str(exc))

    if view is not None:
        by_station = view.set_index("Station")
        total = view["predicted_calls"].sum()
        # The whole city on the left, the stations to watch on the right; the stations picked
        # on the map come further down.
        left, right = st.columns([1, 2])
        left.metric(
            "Citywide expected calls",
            f"{total:.1f}",
            delta=f"{view['difference'].sum():+.1f} vs usual",
            delta_color="off",
        )
        city_low, city_high = citywide_range(total, city_range)
        left.caption(f"Likely range: {city_low} to {city_high}")
        right.markdown("##### Stations to watch")
        watch = stations_to_watch(view)
        if watch.empty:
            right.caption("No station is clearly busier than usual for this shift.")
        for _, row in watch.iterrows():
            right.markdown(
                f"- **{row['Station']}**: likely busier than usual "
                f"(expected {row['predicted_calls']:.1f}, usual {row['usual_calls']:.1f})"
            )
        if not watch.empty:
            right.caption(
                f"Shown when the estimate is at least {WATCH_MIN_EXTRA} calls and "
                f"{WATCH_MIN_SHARE:.0%} above the usual for this weekday. "
                f"At most {WATCH_MAX} are listed."
            )

        st.markdown("##### Expected calls compared with usual, by station area")
        if refused:
            verb = "was" if len(refused) == 1 else "were"
            st.warning(
                f"Up to {MAX_COMPARE} stations can be compared at once, so {', '.join(refused)} "
                f"{verb} not added. Click a picked area again, or remove it from the list below, "
                "to make room."
            )
        areas, boundary = read_station_areas(stamps[0])
        hover = {
            name: {
                "expected": f"{r['predicted_calls']:.1f}",
                "range": f"{int(r['low'])} to {int(r['high'])}",
                "usual": f"{r['usual_calls']:.1f}",
            }
            for name, r in by_station.iterrows()
        }
        outlines = {name: station_color(picked, name, alpha=255) for name in picked}
        # The key keeps the clicked areas when the date, and with it the shading, changes.
        st.pydeck_chart(
            area_map(
                areas,
                boundary,
                stations,
                None,
                fills=difference_fills(view),
                picked_colors=outlines,
                details=hover,
                tooltip_html="<b>{Station}</b><br/>Expected {expected} calls<br/>"
                "Likely range {range}<br/>Usual {usual}",
            ),
            on_select="rerun",
            selection_mode="multi-object",
            key=MAP_KEY,
        )
        st.html(
            f'<div style="height:10px;border-radius:3px;background:linear-gradient(to right,'
            f'{", ".join(DIVERGE_STOPS)})"></div>'
            '<div style="display:flex;justify-content:space-between;font-size:0.8rem">'
            "<span>Quieter than usual</span><span>About usual</span>"
            "<span>Busier than usual</span></div>"
        )
        st.caption(
            "Shaded by expected calls compared with the usual for this weekday (busier or quieter "
            "than usual). This is a forecast, not history. Each area is the part of Seattle "
            "closest to that station. Point at an area for its numbers, click it to pick it."
        )
        st.multiselect(
            f"Compare stations (up to {MAX_COMPARE})",
            station_ids,
            key=PICK_KEY,
            max_selections=MAX_COMPARE,
            placeholder="Choose stations, or click areas on the map",
        )

        if picked:
            st.subheader("Comparing " + ", ".join(picked))
            any_warning = False
            for column, name in zip(st.columns(len(picked)), picked):
                row = by_station.loc[name]
                with column:
                    st.metric(
                        f"{name} expected calls",
                        f"{row['predicted_calls']:.1f}",
                        delta=f"{row['difference']:+.1f} vs usual",
                        delta_color="off",
                    )
                    st.caption(f"Likely range: {int(row['low'])} to {int(row['high'])}")
                    st.caption(f"Usual for this weekday: {row['usual_calls']:.1f}")
                    # Event warnings are informational and do not change the estimate.
                    for warning in get_event_warnings(target_date, name):
                        st.warning(warning)
                        any_warning = True
            if any_warning:
                st.caption("Event warnings are informational and are not included in the forecast.")
        names = picked or [CITY]
        st.markdown("##### Call type mix (average calls per shift)")
        st.dataframe(
            mix_table(mix, names),
            width="stretch",
            column_config={n: st.column_config.NumberColumn(format="%.1f") for n in names},
        )
        st.caption("From past shifts in this season, not a forecast.")

        st.caption(
            "Likely range: about 4 in 5 similar station shifts and about 7 in 10 citywide totals ended up inside it. "
            "It shows how much the estimate can vary and is not a staffing recommendation."
        )
        st.caption(
            "Usual = the average of earlier shifts on this weekday. Difference = expected minus usual."
        )
        st.caption("The estimate does not include weather or events such as games.")
        st.caption(LINKS)
        st.caption("Low and High show the likely range for each station.")
        table = (
            view[TABLE_COLUMNS]
            .sort_values("predicted_calls", ascending=False)
            .rename(columns=COLUMN_NAMES)
        )
        st.dataframe(
            table.style.apply(
                lambda r: [PICK_HIGHLIGHT if r["Station"] in picked else ""] * len(r), axis=1
            ),
            hide_index=True,
            width="stretch",
            # Show one decimal, like the cards above; the full numbers stay in the table.
            column_config={
                COLUMN_NAMES[c]: st.column_config.NumberColumn(format="%.1f")
                for c in ("predicted_calls", "usual_calls", "difference")
            },
        )
        # Create a download table with the counts as shown, the likely range, the target date, and the planning time.
        download = (
            view[TABLE_COLUMNS]
            .round(1)
            .assign(ShiftDate=str(target_date), AsOf=origin.isoformat())
            .rename(columns=COLUMN_NAMES)
        )
        st.download_button(
            "Download displayed counts",
            download.to_csv(index=False),
            f"callcast_counts_{target_date}.csv",
            "text/csv",
            type="primary",
        )
        with st.expander("Hourly breakdown"):
            try:
                # Use hourly_profile.csv shares to divide the shift estimate by hour.
                if picked:
                    hourly = hourly_allocation(
                        ROOT, target_date, picked[0], by_station.loc[picked[0], "predicted_calls"]
                    )
                else:
                    hourly = citywide_hourly(ROOT, target_date, by_station["predicted_calls"])
                st.bar_chart(hourly.set_index("hour")[["expected_calls"]], sort=False)
                st.caption(
                    "Splits the shift total by each hour's usual share of calls, starting at 08:00. "
                    f"Shows {picked[0] if picked else 'the whole city'}: the first station "
                    "picked, or the whole city when none is picked."
                )
            except (OSError, ValueError, KeyError) as exc:
                st.info(str(exc))
    else:
        picked = []
        # The map is gone, and with it its clicked areas.
        st.session_state[CLICKED_KEY] = []
        # Keep the station location map available when no estimate can be produced.
        st.info(
            "No estimate for this shift yet. The station map and history are still below."
        )
        st.map(stations, latitude="lat", longitude="lon", color="#A4282BCC")
        st.caption(LINKS)
    # Plot the first picked station's earlier calls, or the whole city's, from station_shift_counts.csv.
    focus = picked[0] if picked else CITY
    rows = history[history["Station"] == picked[0]] if picked else citywide(history)
    station_history = rows.set_index("ShiftDate")
    st.subheader(f"{focus}: history known at planning time")
    if not station_history.empty:
        recent = station_history[["calls"]].last_valid_index()
        series = station_history[["calls"]].loc[
            pd.Timestamp(recent) - pd.Timedelta(days=90) :
        ]
        # Add every calendar date so outage shifts show up as gaps in the line.
        zoom_safe_line_chart(
            st,
            series["calls"].reindex(
                pd.date_range(series.index.min(), series.index.max(), freq="D")
            ),
        )
        st.caption(
            "Gaps are shifts with missing data. A 0 means there really were no calls."
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
            st.caption(
                note + " (the last one finished by the planning time), without shifts that have "
                "missing data."
            )
    # Three views of the same history: the whole city (default), station areas and
    # comparisons, and past events.
    city_view, stations_view, events_view = st.tabs(["Citywide", "Stations", "Events"])
    with city_view:
        if history.empty:
            st.info("No history is available for this selection.")
        else:
            # One choice drives the map and the charts below: all calls or one call type.
            choice = st.segmented_control(
                "Show calls for", list(CALL_TYPES), default="All calls", key="city_call_type"
            ) or "All calls"
            column, words = CALL_TYPES[choice]
            one = [CITY]
            city = only_type(citywide(history), column)
            # Map: each station area shaded by its average of the chosen calls per shift.
            per_station = history.groupby("Station")[column].mean().reindex(station_ids)
            areas, boundary = read_station_areas(stamps[0])
            st.markdown(f"##### Average {words} per shift by station area")
            st.pydeck_chart(
                value_map(
                    areas,
                    boundary,
                    per_station,
                    {s: f"{v:.1f} {words} per shift" for s, v in per_station.items()},
                ),
                key="city_value_map",
            )
            top = per_station.nlargest(3)
            st.caption(
                f"Lighter areas have fewer {words}, darker areas more: from "
                f"{per_station.min():.1f} ({per_station.idxmin()}) to {per_station.max():.1f} "
                f"per shift. Most: " + ", ".join(f"{s} ({v:.1f})" for s, v in top.items())
                + ". Point at an area for its number."
            )
            st.markdown(f"##### Citywide {words} per shift")
            st.caption("All 34 station areas added together, per shift (08:00 to 08:00).")
            st.dataframe(compare_summary(city, one, average=False), width="stretch")
            st.markdown(f"##### Citywide {words} per shift, last {TREND_DAYS} days (7-shift average)")
            if choice == "All calls":
                # All calls: stacked bands by type, so a rise shows which type caused it.
                st.caption(
                    "Each band is one call type; the top edge is all calls. Each point averages "
                    "the 7 shifts up to that day. Point at the chart to see every type's number."
                )
                figure = stacked_trend_chart(type_trend(citywide(history)))
            else:
                st.caption(
                    "Each point averages the 7 shifts up to that day, which smooths out "
                    "day-to-day swings. Look for lasting rises or drops, not single bumps."
                )
                figure = trend_chart(trend(city, one, average=False))
            st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG)
            a, b = st.columns(2)
            a.markdown(f"##### Average {words} per shift by weekday")
            a.caption("Taller bars are busier days of the week.")
            a.plotly_chart(
                weekday_chart(weekday_means(city, one, average=False)),
                width="stretch",
                config=PLOT_CONFIG,
            )
            b.markdown(f"##### Average {words} per shift by month")
            b.caption("Each point is one month since 2022: shows seasons and longer-term changes.")
            # Averaged per shift, so longer months do not win by size alone.
            zoom_safe_line_chart(b, monthly(city), "%b %Y", color=CITY_COLOR)
            st.markdown("##### Share of calls by category")
            st.caption(
                "How the city's calls split by type (all types, whatever is chosen above). "
                "Point at a bar to see the average number of calls per shift."
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
            st.caption("The Stations tab shows each station area and compares stations.")
    with stations_view:
        # Station area map: city boundary and nearest-station areas, with recent averages on hover.
        st.markdown("##### Station areas")
        picked = []
        if not history.empty:
            areas, boundary = read_station_areas(stamps[0])
            # Shade each area by the share of its calls of one type (shares, not counts, so
            # small and busy stations are on the same scale).
            share_choice = st.segmented_control(
                "Color areas by", list(SHARE_TYPES), default="Fire share", key="station_share_type"
            ) or "Fire share"
            share_column = SHARE_TYPES[share_choice]
            shares, city_share = station_shares(history, share_column)
            share_word = share_choice.split()[0]
            hover = {s: f"{share_word} share: {v:.0f}% (city {city_share:.0f}%)" for s, v in shares.items()}
            # The map is drawn before this run's click is read, so outline the areas picked so far
            # (kept by Streamlit under the chart's key) in the same colours as the charts below.
            before = selected_stations(st.session_state.get("station_area_map"))[:MAX_COMPARE]
            outlines = {name: station_color(before, name, alpha=255) for name in before}
            # Click areas to pick stations; clicking a picked area again unpicks it.
            event = st.pydeck_chart(
                area_map(
                    areas,
                    boundary,
                    stations,
                    recent_averages(history),
                    fills=shade(shares),
                    picked_colors=outlines,
                    extra=hover,
                ),
                on_select="rerun",
                selection_mode="multi-object",
                key="station_area_map",
            )
            picked = selected_stations(event)
            high = shares.nlargest(2)
            st.caption(
                f"Darker areas have a larger share of {share_word.lower()} calls: from "
                f"{shares.min():.0f}% to {shares.max():.0f}% (city {city_share:.0f}%). Highest: "
                + ", ".join(f"{s} ({v:.0f}%)" for s, v in high.items())
                + ". Each area is the part of Seattle closest to that station, so its edges are "
                "approximate. Point at an area for its share and its average calls over the last "
                f"week and month. Click up to {MAX_COMPARE} areas to compare them (outlined in "
                "color); click an area again to remove it."
            )
        if len(picked) > MAX_COMPARE:
            st.warning(
                f"You picked {len(picked)} stations ({', '.join(picked)}). Up to {MAX_COMPARE} can "
                "be compared at once: click an area again to remove it."
            )
        elif picked:
            # 1 to MAX_COMPARE picked stations, each chart with the average station as a reference.
            st.subheader(
                picked[0] if len(picked) == 1 else "Comparing " + ", ".join(picked)
            )
            st.caption(
                f"Average station = citywide calls divided by {len(station_ids)} stations "
                "(grey dashed line), so it is on the same scale as one station."
            )
            st.dataframe(compare_summary(history, picked), width="stretch")
            st.markdown(f"##### Calls per shift, last {TREND_DAYS} days (7-shift average)")
            st.caption(
                "One line per station; the grey dashed line is the average station. Each "
                "point averages the 7 shifts up to that day. Look for lasting rises or drops."
            )
            st.plotly_chart(
                trend_chart(trend(history, picked)), width="stretch", config=PLOT_CONFIG
            )
            st.markdown("##### Average calls per shift by month (all history)")
            st.caption(
                "Each point is one month since 2022: shows seasons and whether a station "
                "has become busier or quieter over the years."
            )
            st.plotly_chart(
                monthly_chart(monthly_means(history, picked)), width="stretch", config=PLOT_CONFIG
            )
            st.markdown("##### Average calls per shift by weekday")
            st.caption("Taller bars are busier days; compare each station with the striped average.")
            st.plotly_chart(
                weekday_chart(weekday_means(history, picked)), width="stretch", config=PLOT_CONFIG
            )
            st.markdown("##### Share of calls by category")
            st.caption(
                "Shares, so busy and quiet stations can be compared directly. Point at a bar "
                "to see the average number of calls per shift too."
            )
            st.plotly_chart(
                category_chart(
                    category_shares(history, picked), category_means(history, picked)
                ),
                width="stretch",
                config=PLOT_CONFIG,
            )
        elif not history.empty:
            st.info(
                f"Click up to {MAX_COMPARE} station areas on the map to see and compare their "
                "history. The Citywide tab shows the whole city."
            )
        if not history.empty:
            # Every station at a glance, in number order; picked stations keep their colours.
            st.markdown("##### Average calls per shift by station")
            st.caption(
                "One bar per station, in number order. Bars above the dashed line are busier "
                "than the average station; picked stations are shown in their colors."
            )
            st.plotly_chart(
                station_bar_chart(station_means(history, station_ids), picked),
                width="stretch",
                config=PLOT_CONFIG,
            )
    with events_view:
        # Past event effects: a venue map over the same station areas, then the tables.
        show_event_effects(st, *read_station_areas(stamps[0]))

with about_tab:
    # Text only: sources and links, what the numbers mean, what is left out, and what they don't show.
    st.write(
        "Source: Seattle Real Time Fire 911 Calls; station locations from Seattle GIS."
    )
    st.markdown(
        "[911 dataset](https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj)"
    )
    st.markdown(LINKS)
    st.markdown("**What the numbers mean**")
    st.write(
        "Each number is the expected count of 911 calls for one station area in one shift, 08:00 to 08:00 the next day. A station area is the area closest to that fire station, so its edges are approximate."
    )
    st.write(
        '"Usual for this weekday" is the average of earlier shifts on the same weekday. Every estimate uses only shifts that had finished by the planning time.'
    )
    st.markdown("**About the likely range**")
    st.write(
        "The likely range comes from how far our estimates were from what really happened across 2025, for stations with a similar expected count. About 4 in 5 of those station shifts ended up inside it, and about 7 in 10 for the citywide total. Some shifts will still land outside it, and it does not say how many units to send."
    )
    st.markdown("**Dates far ahead**")
    st.write(
        "Estimates go up to 90 days after the last data. For a date more than 3 days after the last data, the most recent calls are not known. The estimate then uses the typical values for that station, month and weekday, plus holidays, and is marked as a long-range estimate. The model itself is not retrained, so it does not learn from calls after 2025."
    )
    st.markdown("**What is left out**")
    st.write(
        "Practice calls (TEST), duplicate follow-up records (Fast Back Up), two referral types that stopped being recorded after May 2025 (Triaged Incident and Nurseline/AMR), and shifts where the data feed had a gap."
    )
    st.markdown("**What the numbers don't show**")
    st.write(
        "How long a call takes, which units are free, or travel time. Staffing is the supervisor's decision."
    )
    st.write("This is a fixed set of past calls, not live 911 data.")
