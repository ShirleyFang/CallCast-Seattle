"""Explore > Weather: how citywide calls changed with temperature, cold spells and past storms.

Reads data/weather_effects.csv, data/weather_events.csv and data/weather_event_shifts.csv, made by
scripts/make_weather_effects.py (2022-2025, the Step 15 method; holidays left out of the charts).
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import text_blocks as tb
from chart_theme import (
    AVERAGE_GREY,
    AXIS_TITLE_FONT,
    BORDER_COLOR,
    COLD_BLUE,
    GRID_COLOR,
    PLOT_CONFIG,
    ROSE_LINE,
    SURFACE,
    add_gap_fills,
    apply_plotly_theme,
    day_colors,
    day_gradient,
    hex_to_rgba,
)

ROOT = Path(__file__).resolve().parent
EFFECTS_PATH = ROOT / "data/weather_effects.csv"
EVENTS_PATH = ROOT / "data/weather_events.csv"
EVENT_SHIFTS_PATH = ROOT / "data/weather_event_shifts.csv"

# The event days are shaded a pale cold blue behind the calls line.
EVENT_SHADE = hex_to_rgba(COLD_BLUE, 0.12)
# chart 1 from coldest to hottest (the CSV keeps the order the groups were assigned in)
TEMPERATURE_ORDER = [
    "Low -2 °C or colder",
    "Low -2 to 0 °C",
    "Low 0 to 2 °C",
    "Mild",
    "High 25 to 30 °C",
    "High 30 °C or hotter",
]


def load_weather_effects(
    effects_path=EFFECTS_PATH, events_path=EVENTS_PATH, shifts_path=EVENT_SHIFTS_PATH
):
    """The three CSVs, checked for the columns the charts use."""
    effects = pd.read_csv(effects_path)
    events = pd.read_csv(events_path)
    shifts = pd.read_csv(shifts_path, parse_dates=["ShiftDate"])
    for name, table, needed in [
        (
            "weather_effects.csv",
            effects,
            {
                "chart",
                "group",
                "shifts",
                "events",
                "usual_calls",
                "calls",
                "extra_calls",
                "extra_pct",
                "years",
            },
        ),
        (
            "weather_events.csv",
            events,
            {
                "event",
                "kind",
                "note",
                "lowest_low",
                "highest_high",
                "strongest_gust",
                "usual_calls",
                "calls",
            },
        ),
        (
            "weather_event_shifts.csv",
            shifts,
            {"event", "ShiftDate", "in_event", "calls", "usual_calls"},
        ),
    ]:
        missing = needed - set(table.columns)
        if missing:
            raise ValueError(f"{name} is missing {sorted(missing)}")
    return effects, events, shifts


def _hover(rows, spells=False):
    # A single cold day is not part of a spell, so its bar gives only the day count.
    return [
        f"{r.shifts} days"
        + (f" in {r.events} cold spells" if spells and r.group != "Single cold day" else "")
        + "<br>"
        f"Usual: {r.usual_calls:.0f} calls · on these days: {r.calls:.0f} ({r.extra_pct:+.0f}%)"
        for r in rows.itertuples()
    ]


def _bars(rows, spells=False, **axes):
    # Rounded bars with the extra calls written at the end, like the Events chart. Each bar is
    # on the mint-to-rose scale by its extra calls: none or fewer is mint, the most is rose.
    fills, edges = zip(*(day_colors(v, rows["extra_calls"].max()) for v in rows["extra_calls"]))
    return go.Bar(
        **axes,
        marker={"color": list(fills), "cornerradius": 5, "line": {"color": list(edges), "width": 1}},
        text=[f"{x:+.0f}" for x in rows["extra_calls"]],
        textposition="outside",
        cliponaxis=False,
        customdata=_hover(rows, spells),
        showlegend=False,
    )


def _pad(values):
    # Room past the longest bar for its label.
    low, high = min(float(values.min()), 0.0), max(float(values.max()), 1.0)
    return [low - 0.15 * (high - low) if low < 0 else 0, high * 1.25]


def temperature_table(effects):
    rows = effects[effects["chart"] == "temperature"].set_index("group")
    return rows.loc[[g for g in TEMPERATURE_ORDER if g in rows.index]].reset_index()


def temperature_chart(effects):
    """Horizontal bars, coldest at the top: extra citywide calls per shift against usual."""
    rows = temperature_table(effects)
    fig = go.Figure(
        _bars(
            rows,
            x=rows["extra_calls"],
            y=rows["group"],
            orientation="h",
            hovertemplate="%{x:+.0f} calls a shift<br>%{customdata}<extra></extra>",
        )
    )
    apply_plotly_theme(fig, height=110 + 34 * len(rows))
    fig.add_vline(x=0, line={"color": BORDER_COLOR, "width": 1.5})
    fig.update_layout(
        margin={"l": 10, "r": 40, "t": 12, "b": 32},
        xaxis={
            "title": {"text": "Extra citywide calls per shift vs usual", "font": AXIS_TITLE_FONT},
            "range": _pad(rows["extra_calls"]),
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "griddash": "dot",
        },
        yaxis={"autorange": "reversed", "showgrid": False, "rangemode": "normal"},
    )
    return fig


def spell_table(effects):
    return effects[effects["chart"] == "cold spell"].sort_values("order")


def spell_chart(effects):
    """Bars by day of a cold spell: extra citywide calls per shift against usual."""
    rows = spell_table(effects)
    fig = go.Figure(
        _bars(
            rows,
            spells=True,
            x=rows["group"],
            y=rows["extra_calls"],
            hovertemplate="%{y:+.0f} calls a shift<br>%{customdata}<extra></extra>",
        )
    )
    apply_plotly_theme(fig, y_title="Extra citywide calls per shift", height=300)
    fig.add_hline(y=0, line={"color": BORDER_COLOR, "width": 1.5})
    fig.update_layout(yaxis={"range": _pad(rows["extra_calls"]), "rangemode": "normal"})
    return fig


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
    calls = shifts.set_index("ShiftDate")["calls"]
    fig = go.Figure()
    fig.add_vrect(
        x0=inside.min() - pd.Timedelta(hours=12),
        x1=inside.max() + pd.Timedelta(hours=12),
        fillcolor=EVENT_SHADE,
        line_width=0,
    )
    add_gap_fills(
        fig, calls, hex_to_rgba(ROSE_LINE, 0.13), day_gradient(float(calls.max()) * 1.1)
    )
    fig.add_trace(
        go.Scatter(
            x=shifts["ShiftDate"],
            y=shifts["usual_calls"],
            name="Usual",
            line={"color": AVERAGE_GREY, "dash": "dash", "width": 1.6},
            hovertemplate="Usual: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=calls.index,
            y=calls.to_numpy(),
            name="Calls",
            mode="lines+markers",
            line={"color": ROSE_LINE, "width": 3},
            marker={
                "size": 4.5,
                "color": ROSE_LINE,
                "line": {"width": 1, "color": SURFACE},
            },
            hovertemplate="%{x|%a %b %d}: <b>%{y:.0f}</b> calls<extra></extra>",
        )
    )
    apply_plotly_theme(fig, y_title="Citywide calls per shift", height=320)
    return fig.update_layout(hovermode="x unified", xaxis={"tickformat": "%b %d"})


def show_weather_effects(
    st,
    effects_path=EFFECTS_PATH,
    events_path=EVENTS_PATH,
    shifts_path=EVENT_SHIFTS_PATH,
):
    """Draw the Weather view into a Streamlit container."""
    try:
        effects, events, shifts = load_weather_effects(
            effects_path, events_path, shifts_path
        )
    except (OSError, ValueError) as exc:
        st.info(f"Past weather effects are not available: {exc}")
        return
    first, last = effects["years"].iloc[0].split("-")
    st.html(
        tb.note(
            f"Past days from Jan {first} to Dec {last}, for the whole city. These numbers do not "
            "change with the selected date."
        )
    )

    with st.container(key="explore_card_cold_hot"):
        st.markdown("##### Calls on cold and hot days")
        st.plotly_chart(temperature_chart(effects), width="stretch", config=PLOT_CONFIG)
        st.html(
            tb.block(
                tb.lead(
                    "On very cold or very hot days, the city got noticeably more calls than on a "
                    "usual day."
                ),
                tb.more(
                    "Each bar compares those days with the same weekday in the weeks around them. "
                    "Holidays are left out."
                ),
            )
        )

    with st.container(key="explore_card_cold_spell"):
        st.markdown("##### Calls during a cold spell")
        st.plotly_chart(spell_chart(effects), width="stretch", config=PLOT_CONFIG)
        st.html(
            tb.block(
                tb.lead(
                    "Calls kept rising the longer a cold spell lasted. The first cold day was close "
                    "to usual, and from day 3 the city was clearly busier."
                ),
                tb.terms([("Cold spell", "Two or more days in a row below freezing.")]),
            )
        )

    with st.container(key="explore_card_weather_events"):
        st.markdown("##### Past weather events")
        names = events["event"].tolist()
        choice = (
            st.segmented_control(
                "Weather event", names, default=names[0], key="weather_event"
            )
            or names[0]
        )
        event = events.set_index("event").loc[choice]
        st.markdown(f"**{choice}**: {event_summary(event)}")
        st.plotly_chart(
            event_chart(shifts[shifts["event"] == choice]),
            width="stretch",
            config=PLOT_CONFIG,
        )
        st.html(
            tb.block(
                tb.lead(event["note"]),
                tb.hints(
                    [
                        "{shade} The shaded days are the event.",
                        "{dash} The dashed line is a usual day for that weekday.",
                    ]
                ),
            )
        )
