"""Build the CallCast Seattle page. Run with: uv run streamlit run app.py"""

import pandas as pd
import streamlit as st

from app_data import (
    GROUPS,
    ROOT,
    forecast_origin,
    hourly_allocation,
    known_history,
    learned_predictions,
    load_tables,
    reference_predictions,
    target_dates,
    validate_predictions,
)

st.set_page_config(page_title="CallCast Seattle", layout="wide")
# Hover colors and the hand cursor for clickable parts live in style.css.
st.html(f"<style>{(ROOT / 'style.css').read_text()}</style>")


@st.cache_data
def read_data(stamps):
    # Reuse the loaded tables between screen updates; reread them when a CSV changes.
    return load_tables()


st.title("CallCast Seattle")
st.write(
    "Expected calls per Seattle station area for one shift, 08:00 to 08:00 the next day."
)
try:
    names = ["sfd_stations.csv", "station_shift_counts.csv", "modeling_dataset.csv"]
    # File change times tell the cache when these CSVs need to be read again.
    stamps = tuple((ROOT / "data" / name).stat().st_mtime_ns for name in names)
    # stations: sfd_stations.csv supplies station names and map coordinates.
    # counts: station_shift_counts.csv supplies past call totals and categories.
    # features: modeling_dataset.csv supplies calendar and earlier call inputs for the model.
    stations, counts, features = read_data(stamps)
    # Set the date picker's range from the dates available in the count table.
    min_date, max_date, default_date = target_dates(counts)
except (OSError, ValueError, KeyError) as exc:
    # Stop the page if required CSVs are missing or their contents do not agree.
    st.error(f"Cannot read the shared data: {exc}")
    st.stop()

# Fill the station menu from sfd_stations.csv, ordered by station number.
station_ids = sorted(stations["station"], key=lambda x: int(x.split()[-1]))
first = counts["ShiftDate"].min().date()
last = counts["ShiftDate"].max().date()
st.caption(
    f"Historical demo • data through {last:%b %d, %Y} • {len(station_ids)} station areas"
)
target_date = st.sidebar.date_input(
    "Target shift date",
    value=default_date,
    min_value=min_date,
    max_value=max_date,
)
station = st.sidebar.selectbox(
    "Station area",
    station_ids,
    index=station_ids.index("SFD 10") if "SFD 10" in station_ids else 0,
)
mode = st.sidebar.radio("Prediction source", ["Historical reference", "Trained model"])
# For a Saturday target, show Thursday at 08:00 as the time the supervisor would plan.
origin = forecast_origin(target_date)
st.sidebar.write(f"Planning time: {origin:%a %b %d, %Y at %H:%M %Z}")
st.sidebar.caption(
    "Only uses shifts that had finished by the planning time (Thursday 08:00 for a Saturday shift)."
)

forecast_tab, history_tab, about_tab = st.tabs(
    ["Forecast", "Explore history", "Data and limits"]
)
# Keep only completed earlier shifts from the counts CSV, excluding suspected outages.
history = known_history(counts, target_date)

with forecast_tab:
    end = pd.Timestamp(target_date) + pd.Timedelta(days=1)
    st.subheader(f"{target_date:%a %b %d} 08:00 → {end:%a %b %d} 08:00")
    usual = None
    result = None
    try:
        # Calculate each station's usual calls from its earlier matching weekdays.
        usual = validate_predictions(
            reference_predictions(counts, target_date, station_ids), station_ids
        )
        if mode == "Historical reference":
            # Use that past average as the estimate.
            st.info(
                "Each station's average for this weekday. Trained model not connected yet."
            )
            result, label = usual.copy(), "Historical reference"
        else:
            # Pass modeling_dataset.csv input rows to the model through app_data.py.
            result, label = learned_predictions(features, target_date, station_ids)
            st.caption(f"Model: {label}")
    except (
        OSError,
        ValueError,
        KeyError,
        AttributeError,
        ImportError,
        TypeError,
    ) as exc:
        # Show the model error as a warning.
        st.warning(str(exc))

    if result is not None:
        # Match each station's estimate with its usual count to calculate the difference.
        view = result.merge(
            usual.rename(columns={"predicted_calls": "usual_calls"}),
            on="Station",
            validate="one_to_one",
        )
        view["difference"] = view["predicted_calls"] - view["usual_calls"]
        selected = view.set_index("Station").loc[station]
        # Show all stations added together and the selected station's individual estimate.
        left, right = st.columns(2)
        left.metric("Citywide expected calls", f"{view['predicted_calls'].sum():.1f}")
        right.metric(
            f"{station} expected calls",
            f"{selected['predicted_calls']:.1f}",
            delta=f"{selected['difference']:+.1f} vs usual",
            delta_color="off",
        )
        st.caption(
            "Usual = this weekday's average. In historical reference mode the difference is 0."
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
        st.dataframe(
            view.sort_values("predicted_calls", ascending=False),
            hide_index=True,
            width="stretch",
        )
        # Create a download table with the counts, target date, planning time, and source.
        download = view.assign(
            ShiftDate=str(target_date), AsOf=origin.isoformat(), source=label
        )
        st.download_button(
            "Download displayed counts",
            download.to_csv(index=False),
            "callcast_counts.csv",
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
    # Plot the selected station's earlier calls from station_shift_counts.csv.
    station_history = history.loc[history["Station"] == station].set_index("ShiftDate")
    st.subheader(f"{station}: history known at planning time")
    if not station_history.empty:
        recent = station_history[["calls"]].last_valid_index()
        series = station_history[["calls"]].loc[
            pd.Timestamp(recent) - pd.Timedelta(days=90) :
        ]
        # Add every calendar date so outage shifts show up as gaps in the line.
        st.line_chart(
            series.reindex(
                pd.date_range(series.index.min(), series.index.max(), freq="D")
            )
        )
        st.caption("Gaps are outage shifts. A 0 means the station really had no calls.")

with history_tab:
    # Build comparisons from the same earlier, unmasked counts used in the Forecast tab.
    st.subheader("Patterns before the selected planning time")
    st.caption(
        "Same history as the Forecast tab: finished shifts only, no outage shifts."
    )
    part = history.loc[history["Station"] == station].copy()
    if part.empty:
        st.info("No history is available for this selection.")
    else:
        # Average this station's past calls by weekday, displayed Monday through Sunday.
        part["weekday"] = part["ShiftDate"].dt.dayofweek
        weekday = part.groupby("weekday")["calls"].mean().reindex(range(7))
        weekday.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        a, b = st.columns(2)
        a.write("Average calls by weekday")
        a.bar_chart(weekday, sort=False)
        b.write("Average calls per shift by month")
        # Group by year and month and average per shift so longer months do not win by size alone.
        monthly = part.groupby(part["ShiftDate"].dt.to_period("M"))["calls"].mean()
        monthly.index = monthly.index.astype(str)
        b.line_chart(monthly)
        # Compare every station, then show the selected station's past call category mix.
        st.write("Average calls per shift by station")
        st.bar_chart(history.groupby("Station")["calls"].mean())
        st.write(f"{station}: average calls by category")
        st.bar_chart(part[GROUPS].mean())

with about_tab:
    # Text only: sources and limits.
    st.write(
        "Source: Seattle Real Time Fire 911 Calls; station locations from Seattle GIS."
    )
    st.markdown(
        "[911 dataset](https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj) · [Project guide](https://github.com/ShirleyFang/CallCast-Seattle)"
    )
    st.write(
        "Left out: TEST, Fast Back Up, Triaged Incident and Nurseline/AMR calls, and outage shifts."
    )
    st.write(
        "Counts don't show call length, free units or travel time. The supervisor decides staffing."
    )
    st.write(
        "This replays a fixed dataset, not live 911 data. Model and weather aren't connected yet."
    )
    st.code("uv run streamlit run app.py", language="bash")
