"""Build the CallCast Seattle page. Run with: uv run streamlit run app.py"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_data import (
    ROOT,
    add_likely_range,
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
)
from event_explore import show_event_effects
from station_compare import (
    CITY,
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
    station_bar_chart,
    station_color,
    station_means,
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

# Outside pages a planner can open for live incidents and for station coverage.
# The first 3 shifts after the last day of counts use real recent calls (lag3 is known);
# later shifts use typical values instead (MIN_LAG and data/lag_fill_means.csv in step 4).
RECENT_CALL_SHIFTS = 3

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

# Fill the station menu from sfd_stations.csv, ordered by station number.
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
station = st.sidebar.selectbox(
    "Station area",
    station_ids,
    index=station_ids.index("SFD 10") if "SFD 10" in station_ids else 0,
)

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
    usual = None
    result = None
    try:
        # Calculate each station's usual calls from earlier matching weekdays.
        usual = validate_predictions(
            reference_predictions(counts, target_date, station_ids), station_ids
        )

        # Always use the trained model for the supervisor-facing forecast.
        result, _ = learned_predictions(features, target_date, station_ids)
    except (
        OSError,
        ValueError,
        KeyError,
        AttributeError,
        ImportError,
        TypeError,
    ) as exc:
        st.warning(str(exc))

    if result is not None:
        # Match each station's estimate with its usual count to calculate the difference.
        view = result.merge(
            usual.rename(columns={"predicted_calls": "usual_calls"}),
            on="Station",
            validate="one_to_one",
        )
        view["difference"] = view["predicted_calls"] - view["usual_calls"]
        # Low and high: the estimate plus the usual misses for a forecast of that size.
        view = add_likely_range(view, bands)
        selected = view.set_index("Station").loc[station]
        # Show all stations added together and the selected station's individual estimate.
        left, right = st.columns(2)
        left.metric(
            "Citywide expected calls",
            f"{view['predicted_calls'].sum():.1f}",
            delta=f"{view['difference'].sum():+.1f} vs usual",
            delta_color="off",
        )
        right.metric(
            f"{station} expected calls",
            f"{selected['predicted_calls']:.1f}",
            delta=f"{selected['difference']:+.1f} vs usual",
            delta_color="off",
        )
        city_low, city_high = citywide_range(view["predicted_calls"].sum(), city_range)
        left.caption(f"Likely range: {city_low} to {city_high}")
        right.caption(
            f"Likely range: {int(selected['low'])} to {int(selected['high'])}"
        )
        st.caption(
            "Likely range: about 4 in 5 similar past shifts ended up inside it. "
            "It shows how much the estimate can vary and is not a staffing recommendation."
        )
        st.caption(
            "Usual = the average of earlier shifts on this weekday. Difference = expected minus usual."
        )
        st.caption(
            "The estimate does not include weather or events such as games."
        )
        # Add lat/lon from sfd_stations.csv to put each station's estimate on the map.
        map_data = stations.merge(
            view, left_on="station", right_on="Station", validate="one_to_one"
        )
        # Bigger counts get bigger points; adding 1 keeps stations with zero calls visible.
        map_data["point_size"] = (map_data["predicted_calls"] + 1) * 12
        st.map(
            map_data,
            latitude="lat",
            longitude="lon",
            size="point_size",
            color="#A4282BCC",
        )
        st.caption("Dots are fire stations. Bigger dots mean more expected calls.")
        st.caption(LINKS)
        st.caption("Low and High show the likely range for each station.")
        st.dataframe(
            view[TABLE_COLUMNS]
            .sort_values("predicted_calls", ascending=False)
            .rename(columns=COLUMN_NAMES),
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
                # Use hourly_profile.csv shares to divide this station's shift estimate by hour.
                hourly = hourly_allocation(
                    ROOT, target_date, station, selected["predicted_calls"]
                )
                st.bar_chart(hourly.set_index("hour")[["expected_calls"]], sort=False)
                st.caption(
                    "Splits the shift total by each hour's usual share of calls, starting at 08:00."
                )
            except (OSError, ValueError, KeyError) as exc:
                st.info(str(exc))
    else:
        # Keep the station location map available when no estimate can be produced.
        st.info(
            "No estimate for this shift yet. The station map and history are still below."
        )
        st.map(stations, latitude="lat", longitude="lon", color="#A4282BCC")
        st.caption(LINKS)
    # Plot the selected station's earlier calls from station_shift_counts.csv.
    station_history = history.loc[history["Station"] == station].set_index("ShiftDate")
    st.subheader(f"{station}: history known at planning time")
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
            "Gaps are shifts with missing data. A 0 means the station really had no calls."
        )

with explore_tab:
    # Build comparisons from the same earlier, unmasked counts used in the Prediction tab.
    st.subheader("Patterns before the selected planning time")
    st.caption(
        "Same history as the Prediction tab: finished shifts only, without shifts that have missing data."
    )
    # Say which shifts the charts use: they stop at the planning time, and dates past the
    # data all use the same, latest history.
    if not history.empty:
        through = history["ShiftDate"].max()
        note = f"Citywide and Stations use finished shifts up to {through:%a %b %d, %Y}"
        # Compare with the last day of counts, not `through`: an outage shift just before the
        # planning time also makes `through` earlier, but newer data still exists.
        if target_date - pd.Timedelta(days=3) > last:
            note += (
                ", the last day of data. Later dates show the same history until "
                "new calls are added."
            )
        else:
            note += ", the last shift finished by the planning time."
        st.caption(note)
    # Three views of the same history: the whole city (default), station areas and
    # comparisons, and past events.
    city_view, stations_view, events_view = st.tabs(["Citywide", "Stations", "Events"])
    with city_view:
        if history.empty:
            st.info("No history is available for this selection.")
        else:
            city = citywide(history)
            one = [CITY]
            st.caption("All 34 station areas added together, per shift (08:00 to 08:00).")
            st.dataframe(compare_summary(city, one, average=False), width="stretch")
            st.write(f"Citywide calls per shift, last {TREND_DAYS} days (7-shift average)")
            st.caption(
                "Each point averages the 7 shifts up to that day, which smooths out "
                "day-to-day swings. Look for lasting rises or drops, not single bumps."
            )
            st.plotly_chart(
                trend_chart(trend(city, one, average=False)), width="stretch", config=PLOT_CONFIG
            )
            a, b = st.columns(2)
            a.write("Average calls per shift by weekday")
            a.caption("Taller bars are busier days of the week.")
            a.plotly_chart(
                weekday_chart(weekday_means(city, one, average=False)),
                width="stretch",
                config=PLOT_CONFIG,
            )
            b.write("Average calls per shift by month")
            b.caption("Each point is one month since 2022: shows seasons and longer-term changes.")
            # Averaged per shift, so longer months do not win by size alone.
            zoom_safe_line_chart(b, monthly(city), "%b %Y", color=CITY_COLOR)
            st.write("Share of calls by category")
            st.caption(
                "How the city's calls split by type. Point at a bar to see the average "
                "number of calls per shift."
            )
            st.plotly_chart(
                category_chart(
                    category_shares(city, one, average=False),
                    category_means(city, one, average=False),
                ),
                width="stretch",
                config=PLOT_CONFIG,
            )
            st.caption("The Stations tab shows each station area and compares stations.")
    with stations_view:
        # Station area map: city boundary and nearest-station areas, with recent averages on hover.
        st.write("Station areas")
        picked = []
        if not history.empty:
            areas, boundary = read_station_areas(stamps[0])
            # The map is drawn before this run's click is read, so colour the areas picked so far
            # (kept by Streamlit under the chart's key) with the same colours as the charts below.
            before = selected_stations(st.session_state.get("station_area_map"))[:MAX_COMPARE]
            fills = {name: station_color(before, name, alpha=150) for name in before}
            # Click areas to pick stations; clicking a picked area again unpicks it.
            event = st.pydeck_chart(
                area_map(areas, boundary, stations, recent_averages(history), fills),
                on_select="rerun",
                selection_mode="multi-object",
                key="station_area_map",
            )
            picked = selected_stations(event)
            st.caption(
                "Each outlined area is the part of Seattle closest to that station, so its edges are "
                "approximate. Point at an area to see its average calls per shift over the last "
                f"week and month known at the planning time. Click up to {MAX_COMPARE} areas to "
                "compare them; click an area again to remove it."
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
            st.write(f"Calls per shift, last {TREND_DAYS} days (7-shift average)")
            st.caption(
                "One line per station; the grey dashed line is the average station. Each "
                "point averages the 7 shifts up to that day. Look for lasting rises or drops."
            )
            st.plotly_chart(
                trend_chart(trend(history, picked)), width="stretch", config=PLOT_CONFIG
            )
            st.write("Average calls per shift by month (all history)")
            st.caption(
                "Each point is one month since 2022: shows seasons and whether a station "
                "has become busier or quieter over the years."
            )
            st.plotly_chart(
                monthly_chart(monthly_means(history, picked)), width="stretch", config=PLOT_CONFIG
            )
            st.write("Average calls per shift by weekday")
            st.caption("Taller bars are busier days; compare each station with the striped average.")
            st.plotly_chart(
                weekday_chart(weekday_means(history, picked)), width="stretch", config=PLOT_CONFIG
            )
            st.write("Share of calls by category")
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
            st.write("Average calls per shift by station")
            st.caption(
                "One bar per station, in number order. Bars above the dashed line are busier "
                "than the average station; picked stations are shown in their colours."
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
        "The likely range comes from how far our estimates were from what really happened across 2025, for stations with a similar expected count. About 4 in 5 of those shifts ended up inside it. Some shifts will still land outside it, and it does not say how many units to send."
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
