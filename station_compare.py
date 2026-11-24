"""Compare up to MAX_COMPARE stations picked on the Explore map, next to the average station.

Every number comes from the app's known_history(): finished shifts by the planning time, with
no missing-data shifts. The "average station" is the citywide total divided by the number of
stations, so it sits on the same scale as one station.
"""

import pandas as pd
import plotly.graph_objects as go

MAX_COMPARE = 3
AVERAGE = "Average station"
GROUPS = ["fire", "medical", "non_emergency", "other"]
GROUP_NAMES = {"fire": "Fire", "medical": "Medical", "non_emergency": "Non-emergency", "other": "Other"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Same colours as chartCategoricalColors in .streamlit/config.toml; the average is grey, dashed.
COLORS = ["#2F5D8A", "#A4282B", "#C99A3B"]
AVERAGE_COLOR = "#6B7280"
# Category colours stay clear of the station colours, so "red" always means one station.
CATEGORY_COLORS = {"Fire": "#E07A5F", "Medical": "#3D405B", "Non-emergency": "#81B29A", "Other": "#D9C9A3"}
PLOT_CONFIG = {"displayModeBar": False}  # same as the app's other charts: no plotly toolbar
TREND_DAYS = 90
ROLLING = 7
CITY = "Citywide"  # the whole city's total, for the Explore > Citywide view


def station_color(stations, name, alpha=None):
    """The chart colour of a picked station, as hex or as an RGBA list for the map."""
    hex_color = COLORS[list(stations).index(name) % len(COLORS)]
    if alpha is None:
        return hex_color
    return [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)] + [alpha]


def average_station(history):
    """One row per shift: the citywide counts divided by the number of stations."""
    n = history["Station"].nunique()
    city = history.groupby("ShiftDate")[["calls", *GROUPS]].sum() / n
    return city.reset_index().assign(Station=AVERAGE)


def citywide(history):
    """One row per shift: the citywide total of calls and of each category.

    Missing-data shifts are dropped for every station at once, so each total covers all
    stations.
    """
    city = history.groupby("ShiftDate")[["calls", *GROUPS]].sum()
    return city.reset_index().assign(Station=CITY)


def with_average(history, stations, average=True):
    """The picked stations' rows, plus the average station's rows if average is True."""
    picked = history[history["Station"].isin(stations)]
    if not average:
        return picked
    return pd.concat([picked, average_station(history)], ignore_index=True)


def _lines(stations, average):
    return [*stations, AVERAGE] if average else list(stations)


def trend(history, stations, days=TREND_DAYS, window=ROLLING, average=True):
    """7-shift rolling mean of calls over the last `days` known shifts, one column per line.

    Missing-data shifts stay as gaps in the dates and are skipped by the rolling mean.
    """
    rows = with_average(history, stations, average)
    latest = rows["ShiftDate"].max()
    wide = rows.pivot_table(index="ShiftDate", columns="Station", values="calls")
    full = pd.date_range(latest - pd.Timedelta(days=days + window - 2), latest)
    smooth = wide.reindex(full).rolling(window, min_periods=4).mean()
    return smooth.loc[latest - pd.Timedelta(days=days - 1):, _lines(stations, average)]


def weekday_means(history, stations, average=True):
    """Average calls per shift by weekday (Mon-Sun rows), one column per line."""
    rows = with_average(history, stations, average)
    rows = rows.assign(weekday=rows["ShiftDate"].dt.dayofweek)
    table = rows.pivot_table(index="weekday", columns="Station", values="calls", aggfunc="mean")
    return table.reindex(range(7)).set_axis(WEEKDAYS)[_lines(stations, average)]


def category_shares(history, stations, average=True):
    """Percent of each station's calls in each category (rows: stations, columns: categories)."""
    rows = with_average(history, stations, average)
    totals = rows.groupby("Station")[GROUPS].sum()
    shares = totals.div(totals.sum(axis=1), axis=0) * 100
    return shares.loc[_lines(stations, average)].rename(columns=GROUP_NAMES)


def summary(history, stations, average=True):
    """Last week, last month and all-time averages, and the busiest weekday, per station."""
    rows = with_average(history, stations, average)
    latest = rows["ShiftDate"].max()
    out = {}
    for name, days in [("Last week", 7), ("Last month", 28)]:
        recent = rows[rows["ShiftDate"] > latest - pd.Timedelta(days=days)]
        out[name] = recent.groupby("Station")["calls"].mean()
    out["All history"] = rows.groupby("Station")["calls"].mean()
    out["Busiest weekday"] = weekday_means(history, stations, average).idxmax()
    table = pd.DataFrame(out).loc[_lines(stations, average)]
    return table.round({"Last week": 1, "Last month": 1, "All history": 1})


def monthly(rows):
    """Average calls per shift by month, for one line (e.g. the citywide rows)."""
    out = rows.groupby(rows["ShiftDate"].dt.to_period("M"))["calls"].mean()
    out.index = out.index.to_timestamp()
    return out


def monthly_means(history, stations, average=True):
    """Average calls per shift by month (rows: months), one column per line.

    Averaged per shift, so longer months do not win by size alone.
    """
    rows = with_average(history, stations, average)
    month = rows["ShiftDate"].dt.to_period("M").dt.to_timestamp()
    table = rows.assign(month=month).pivot_table(index="month", columns="Station", values="calls", aggfunc="mean")
    return table[_lines(stations, average)]


def category_means(history, stations, average=True):
    """Average calls per shift in each category (rows: stations, columns: categories)."""
    rows = with_average(history, stations, average)
    means = rows.groupby("Station")[GROUPS].mean()
    return means.loc[_lines(stations, average)].rename(columns=GROUP_NAMES)


def _style(name, i):
    if name == AVERAGE:
        return {"color": AVERAGE_COLOR, "dash": "dash"}
    if name == CITY:
        return {"color": CITY_COLOR}
    return {"color": COLORS[i % len(COLORS)]}


def _layout(fig, y_title, height=320):
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1F2A36"},
        legend={"orientation": "h", "y": -0.18},
        yaxis={"title": y_title, "gridcolor": "#E6E1D9", "rangemode": "tozero"},
        xaxis={"gridcolor": "#E6E1D9"},
    )
    return fig


def trend_chart(table):
    fig = go.Figure()
    for i, name in enumerate(table.columns):
        style = _style(name, i)
        fig.add_trace(go.Scatter(
            x=table.index, y=table[name], name=name, mode="lines",
            line={"width": 2, **style}, connectgaps=False,
            hovertemplate="%{x|%b %d}: %{y:.1f} calls<extra>" + name + "</extra>",
        ))
    return _layout(fig, "Calls per shift (7-shift average)")


def monthly_chart(table):
    """Long-term view: average calls per shift in each month since the data starts."""
    fig = go.Figure()
    for i, name in enumerate(table.columns):
        style = _style(name, i)
        fig.add_trace(go.Scatter(
            x=table.index, y=table[name], name=name, mode="lines",
            line={"width": 2, **style}, connectgaps=False,
            hovertemplate="%{x|%b %Y}: %{y:.1f} calls per shift<extra>" + name + "</extra>",
        ))
    return _layout(fig, "Average calls per shift")


def weekday_chart(table):
    fig = go.Figure()
    for i, name in enumerate(table.columns):
        style = _style(name, i)
        fig.add_trace(go.Bar(
            x=table.index, y=table[name], name=name, marker={"color": style["color"]},
            marker_pattern_shape="/" if name == AVERAGE else "",
            hovertemplate="%{x}: %{y:.1f} calls<extra>" + name + "</extra>",
        ))
    return _layout(fig.update_layout(barmode="group"), "Average calls per shift")


# Neutral slate for "the whole city / stations not picked", so colour always means a picked
# station: lighter for the all-stations bars, darker for citywide lines and bars.
BAR_COLOR = "#8C97A4"
CITY_COLOR = "#5F6E7F"


def station_means(history, station_ids):
    """Average calls per shift for every station, in station number order."""
    return history.groupby("Station")["calls"].mean().reindex(station_ids)


def station_bar_chart(means, picked=()):
    """All stations' averages in the theme green; picked stations in their comparison colours;
    the average station as a grey dashed line."""
    picked = [s for s in picked if s in means.index][:MAX_COMPARE]
    colors = [station_color(picked, s) if s in picked else BAR_COLOR for s in means.index]
    fig = go.Figure(go.Bar(
        x=list(means.index), y=means.to_numpy(), marker={"color": colors},
        hovertemplate="%{x}: %{y:.1f} calls per shift<extra></extra>", showlegend=False,
    ))
    average = means.mean()
    fig.add_hline(
        y=average, line={"color": AVERAGE_COLOR, "dash": "dash", "width": 1.5},
        annotation_text=f"Average station {average:.1f}", annotation_position="top right",
        annotation_font={"color": AVERAGE_COLOR, "size": 12},
    )
    fig = _layout(fig, "Average calls per shift", height=340)
    fig.update_xaxes(tickangle=-90, tickfont={"size": 11})
    return fig


def category_chart(table, counts=None):
    """100% bars: one bar per station, split by category.

    counts (same shape, from category_means) adds the average calls per shift to the hover
    text, so the share and the actual number are both visible.
    """
    fig = go.Figure()
    for category in table.columns:
        hover = "%{y}: %{x:.0f}% " + category.lower()
        extra = {}
        if counts is not None:
            extra["customdata"] = counts[category].to_numpy()
            hover += " (about %{customdata:.1f} calls per shift)"
        fig.add_trace(go.Bar(
            y=table.index, x=table[category], name=category, orientation="h",
            marker={"color": CATEGORY_COLORS[category]},
            hovertemplate=hover + "<extra></extra>", **extra,
        ))
    fig = _layout(fig, "", height=140 + 45 * len(table))
    fig.update_layout(
        barmode="stack",
        xaxis={"title": "Share of calls (%)", "range": [0, 100], "gridcolor": "#E6E1D9"},
        legend={"orientation": "h", "y": -0.35, "traceorder": "normal"},
    )
    fig.update_yaxes(autorange="reversed")
    return fig
