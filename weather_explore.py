"""Explore > Weather: how citywide calls changed with temperature, cold spells and past storms.

Reads data/weather_effects.csv, data/weather_events.csv and data/weather_event_shifts.csv, made by
scripts/make_weather_effects.py (2022-2025, the Step 15 method; holidays left out of the charts).
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from station_compare import AVERAGE_COLOR, CITY_COLOR, PLOT_CONFIG

ROOT = Path(__file__).resolve().parent
EFFECTS_PATH = ROOT / "data/weather_effects.csv"
EVENTS_PATH = ROOT / "data/weather_events.csv"
EVENT_SHIFTS_PATH = ROOT / "data/weather_event_shifts.csv"

COLD, MILD, HOT = "#2F5D8A", "#8C97A4", "#C9733B"
EVENT_SHADE = "rgba(47, 93, 138, 0.12)"
# chart 1 from coldest to hottest (the CSV keeps the order the groups were assigned in)
TEMPERATURE_ORDER = ["Low -2 °C or colder", "Low -2 to 0 °C", "Low 0 to 2 °C", "Mild",
                     "High 25 to 30 °C", "High 30 °C or hotter"]


def load_weather_effects(effects_path=EFFECTS_PATH, events_path=EVENTS_PATH, shifts_path=EVENT_SHIFTS_PATH):
    """The three CSVs, checked for the columns the charts use."""
    effects = pd.read_csv(effects_path)
    events = pd.read_csv(events_path)
    shifts = pd.read_csv(shifts_path, parse_dates=["ShiftDate"])
    for name, table, needed in [
        ("weather_effects.csv", effects, {"chart", "group", "shifts", "events", "usual_calls", "calls",
                                          "extra_calls", "extra_pct", "years"}),
        ("weather_events.csv", events, {"event", "kind", "note", "lowest_low", "highest_high",
                                        "strongest_gust", "usual_calls", "calls"}),
        ("weather_event_shifts.csv", shifts, {"event", "ShiftDate", "in_event", "calls", "usual_calls"}),
    ]:
        missing = needed - set(table.columns)
        if missing:
            raise ValueError(f"{name} is missing {sorted(missing)}")
    return effects, events, shifts


def _layout(fig, height=300):
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1F2A36"},
        showlegend=False,
    )
    return fig


def _hover(rows, spells=False):
    return [
        f"{r.shifts} days" + (f" in {r.events} cold spells" if spells else "") + "<br>"
        f"Usual: {r.usual_calls:.0f} calls · on these days: {r.calls:.0f} ({r.extra_pct:+.0f}%)"
        for r in rows.itertuples()
    ]


def temperature_table(effects):
    rows = effects[effects["chart"] == "temperature"].set_index("group")
    return rows.loc[[g for g in TEMPERATURE_ORDER if g in rows.index]].reset_index()


def temperature_chart(effects):
    """Horizontal bars, coldest at the top: extra citywide calls per shift against usual."""
    rows = temperature_table(effects)
    colors = [COLD if g.startswith("Low") else HOT if g.startswith("High") else MILD for g in rows["group"]]
    fig = go.Figure(go.Bar(
        x=rows["extra_calls"], y=rows["group"], orientation="h", marker={"color": colors},
        customdata=_hover(rows), hovertemplate="%{x:+.0f} calls a shift<br>%{customdata}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="Extra citywide calls per shift vs usual", zeroline=True,
                     zerolinecolor=AVERAGE_COLOR, gridcolor="#E6E1D9")
    return _layout(fig)


def spell_table(effects):
    return effects[effects["chart"] == "cold spell"].sort_values("order")


def spell_chart(effects):
    """Bars by day of a cold spell: extra citywide calls per shift against usual."""
    rows = spell_table(effects)
    colors = [MILD if g == "Single cold day" else COLD for g in rows["group"]]
    fig = go.Figure(go.Bar(
        x=rows["group"], y=rows["extra_calls"], marker={"color": colors},
        customdata=_hover(rows, spells=True),
        hovertemplate="%{y:+.0f} calls a shift<br>%{customdata}<extra></extra>",
    ))
    fig.update_yaxes(title="Extra citywide calls per shift", zeroline=True, zerolinecolor=AVERAGE_COLOR,
                     gridcolor="#E6E1D9")
    fig.update_xaxes(title=None)
    return _layout(fig)


def event_summary(event):
    """One line about an event, e.g. 'Lowest low -10 °C · about 459 calls a shift vs about 302 usual'."""
    if event["kind"] == "heat":
        weather = f"Highest high {event['highest_high']:.0f} °C"
    elif event["kind"] == "wind":
        weather = f"Strongest gust {event['strongest_gust']:.0f} km/h"
    else:
        weather = f"Lowest low {event['lowest_low']:.0f} °C"
    return f"{weather} · about {event['calls']:.0f} calls a shift vs about {event['usual_calls']:.0f} usual"


def event_chart(shifts):
    """Citywide calls each day around one event, the usual level dashed, event days shaded."""
    shifts = shifts.sort_values("ShiftDate")
    inside = shifts.loc[shifts["in_event"], "ShiftDate"]
    fig = go.Figure()
    fig.add_vrect(x0=inside.min() - pd.Timedelta(hours=12), x1=inside.max() + pd.Timedelta(hours=12),
                  fillcolor=EVENT_SHADE, line_width=0)
    fig.add_trace(go.Scatter(x=shifts["ShiftDate"], y=shifts["usual_calls"], name="Usual",
                             line={"color": AVERAGE_COLOR, "dash": "dash", "width": 1.5},
                             hovertemplate="Usual: %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=shifts["ShiftDate"], y=shifts["calls"], name="Calls", mode="lines+markers",
                             line={"color": CITY_COLOR, "width": 2},
                             hovertemplate="%{x|%a %b %d}: %{y:.0f} calls<extra></extra>"))
    fig.update_yaxes(title="Citywide calls per shift", gridcolor="#E6E1D9", rangemode="tozero")
    fig.update_xaxes(tickformat="%b %d", gridcolor="#E6E1D9")
    return _layout(fig.update_layout(hovermode="x unified"), height=320)


def show_weather_effects(st, effects_path=EFFECTS_PATH, events_path=EVENTS_PATH, shifts_path=EVENT_SHIFTS_PATH):
    """Draw the Weather view into a Streamlit container."""
    try:
        effects, events, shifts = load_weather_effects(effects_path, events_path, shifts_path)
    except (OSError, ValueError) as exc:
        st.info(f"Past weather effects are not available: {exc}")
        return
    first, last = effects["years"].iloc[0].split("-")
    st.caption(
        f"Past days from Jan {first} to Dec {last}, for the whole city. These numbers do not change "
        "with the selected date."
    )

    st.markdown("##### Calls on cold and hot days")
    st.plotly_chart(temperature_chart(effects), width="stretch", config=PLOT_CONFIG)
    st.caption(
        "On very cold or very hot days, the city got noticeably more calls than on a usual day. "
        "Each bar compares those days with the same weekday in the weeks around them. "
        "Holidays are left out."
    )

    st.markdown("##### Calls during a cold spell")
    st.plotly_chart(spell_chart(effects), width="stretch", config=PLOT_CONFIG)
    st.caption(
        "Calls kept rising the longer a cold spell lasted. The first cold day was close to usual, "
        "and from day 3 the city was clearly busier. A cold spell is two or more days in a row "
        "below freezing."
    )

    st.markdown("##### Past weather events")
    names = events["event"].tolist()
    choice = st.segmented_control("Weather event", names, default=names[0], key="weather_event") or names[0]
    event = events.set_index("event").loc[choice]
    st.markdown(f"**{choice}**: {event_summary(event)}")
    st.plotly_chart(event_chart(shifts[shifts["event"] == choice]), width="stretch", config=PLOT_CONFIG)
    st.caption(
        f"{event['note']} The shaded days are the event. The dashed line is a usual day for that weekday."
    )
