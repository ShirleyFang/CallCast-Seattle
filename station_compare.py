"""Compare up to MAX_COMPARE stations picked on the Explore map, next to the average station.

Every number comes from the app's known_history(): finished shifts by the planning time, with
no missing-data shifts. The "average station" is the citywide total divided by the number of
stations, so it sits on the same scale as one station.
"""

import pandas as pd
import plotly.graph_objects as go

from chart_theme import (
    AVERAGE_GREY,
    AXIS_TITLE_FONT,
    BORDER_COLOR,
    CATEGORY_COLORS,
    CATEGORY_EDGES,
    CATEGORY_TEXT,
    FONT_FAMILY,
    FS_XS,
    GRID_COLOR,
    INK,
    MUTED_INK,
    ROSE_LINE,
    STATION_COLORS,
    STATION_EDGES,
    SURFACE,
    add_gap_fills,
    apply_plotly_theme,
    day_colors,
    day_gradient,
    hex_to_rgba,
    rgb,
)

MAX_COMPARE = 3
AVERAGE = "Average station"
GROUPS = ["fire", "medical", "non_emergency", "other"]
GROUP_NAMES = {"fire": "Fire", "medical": "Medical", "non_emergency": "Non-emergency", "other": "Other"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Same colours as chartCategoricalColors in .streamlit/config.toml; the average is grey, dashed.
# The city alone is a deep rose line over the mint-to-rose fill, its bars on that same scale.
COLORS = STATION_COLORS
AVERAGE_COLOR = AVERAGE_GREY
CITY_COLOR = ROSE_LINE
TREND_DAYS = 90
ROLLING = 7
CITY = "Citywide"  # the whole city's total, for the Explore > Citywide view
# Explore > Citywide "Show calls for": button label -> (count column, words for titles)
CALL_TYPES = {
    "All calls": ("calls", "calls"),
    "Medical": ("medical", "medical calls"),
    "Fire": ("fire", "fire calls"),
    "Non-emergency": ("non_emergency", "non-emergency calls"),
    "Other": ("other", "other calls"),
}


# Explore > Stations "Colour areas by": how much of each station's calls are of one type.
# Shares, not counts: busy and quiet stations can then be compared on the same scale.
SHARE_TYPES = {"Fire share": "fire", "Medical share": "medical", "Other share": "other"}


def station_shares(history, column):
    """Percent of each station's calls of one type, and the same percent for the whole city."""
    totals = history.groupby("Station")[[column, "calls"]].sum()
    share = totals[column] / totals["calls"] * 100
    city = totals[column].sum() / totals["calls"].sum() * 100
    return share, city


def only_type(rows, column):
    """The same rows with `calls` replaced by one call type, so every table and chart that
    counts `calls` now counts that type."""
    return rows.assign(calls=rows[column]) if column != "calls" else rows


def station_color(stations, name, alpha=None):
    """The chart colour of a picked station, as hex or as an RGBA list for the map."""
    hex_color = COLORS[list(stations).index(name) % len(COLORS)]
    return hex_color if alpha is None else rgb(hex_color, alpha)


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


def _layout(fig, y_title, height=320, show_legend=True, legend_top=True):
    return apply_plotly_theme(
        fig, y_title=y_title, height=height, show_legend=show_legend, legend_top=legend_top
    )


def trend_chart(table):
    fig = go.Figure()
    single = len(table.columns) == 1
    for i, name in enumerate(table.columns):
        style = _style(name, i)
        trace = {
            "x": table.index,
            "y": table[name],
            "name": name,
            "mode": "lines",
            "line": {"width": 1.8 if name == AVERAGE else 3 if single else 2.5, **style},
            "connectgaps": False,
            "hovertemplate": "%{x|%b %d}: <b>%{y:.1f}</b> calls<extra>" + name + "</extra>",
        }
        if single:
            top = float(table[name].max()) * 1.1
            add_gap_fills(
                fig, table[name], hex_to_rgba(style["color"], 0.14), day_gradient(top)
            )
        fig.add_trace(go.Scatter(**trace))
    if single:
        avg_val = float(table.iloc[:, 0].mean())
        if pd.notna(avg_val):
            fig.add_hline(
                y=avg_val,
                line={"color": MUTED_INK, "dash": "dot", "width": 1.25},
                annotation_text=f"Avg {avg_val:.1f}",
                annotation_position="top right",
                annotation_font={"color": MUTED_INK, "size": FS_XS, "family": FONT_FAMILY},
            )
    return _layout(fig, "Calls per shift (7-shift average)", show_legend=not single)


STACK_ORDER = ["medical", "fire", "other", "non_emergency"]  # bottom to top: biggest first


def type_trend(city_rows, days=TREND_DAYS, window=ROLLING):
    """7-shift rolling mean of each call type over the last `days` shifts (citywide rows).

    Same window and gaps as trend(), one column per call type, so the bands add up to the
    all-calls line.
    """
    rows = city_rows.set_index("ShiftDate")
    latest = rows.index.max()
    full = pd.date_range(latest - pd.Timedelta(days=days + window - 2), latest)
    smooth = rows[STACK_ORDER].reindex(full).rolling(window, min_periods=4).mean()
    return smooth.loc[latest - pd.Timedelta(days=days - 1):].rename(columns=GROUP_NAMES)


def stacked_trend_chart(table):
    """Citywide calls by type as stacked bands; the top edge is all calls."""
    # Stacked areas draw a missing value as 0, which would show a false drop to zero (e.g. on
    # a day without enough shifts for the 7-shift average), so those days are left out.
    table = table.dropna(how="any")
    fig = go.Figure()
    for name in table.columns:
        color = CATEGORY_COLORS[name]
        fig.add_trace(go.Scatter(
            x=table.index, y=table[name], name=name, mode="lines", stackgroup="calls",
            line={"width": 1.2, "color": CATEGORY_EDGES[name]}, fillcolor=hex_to_rgba(color, 0.78),
            hovertemplate="%{y:.0f}<extra>" + name + "</extra>",
        ))
    total = table.sum(axis=1, min_count=1)
    fig.add_trace(go.Scatter(  # crisp outline on top, so the all-calls total shows on hover
        x=table.index, y=total, name="All calls", mode="lines", showlegend=False,
        line={"width": 2, "color": INK}, hovertemplate="%{y:.0f}<extra>All calls</extra>",
    ))
    fig = _layout(fig, "Calls per shift (7-shift average)", height=340, show_legend=True, legend_top=True)
    fig.update_layout(
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.14, "yanchor": "top", "x": 0, "traceorder": "reversed"},
    )
    return fig


def monthly_chart(table):
    """Long-term view: average calls per shift in each month since the data starts."""
    fig = go.Figure()
    single = len(table.columns) == 1
    for i, name in enumerate(table.columns):
        style = _style(name, i)
        fig.add_trace(go.Scatter(
            x=table.index, y=table[name], name=name, mode="lines+markers",
            line={"width": 1.8 if name == AVERAGE else 2.3, **style},
            marker={"size": 4, "color": style["color"]},
            connectgaps=False,
            hovertemplate="%{x|%b %Y}: <b>%{y:.1f}</b> calls per shift<extra>" + name + "</extra>",
        ))
    return _layout(fig, "Average calls per shift", show_legend=not single)


def weekday_chart(table):
    fig = go.Figure()
    single = len(table.columns) == 1
    for i, name in enumerate(table.columns):
        style = _style(name, i)
        marker = {"color": style["color"], "cornerradius": 6}
        if single:
            # The city alone: each weekday on the mint-to-rose scale, from the quietest day
            # (mint) to the busiest (rose), so close averages still look different.
            col = table[name]
            fills, edges = zip(*(day_colors(v, col.max(), col.min()) for v in col))
            marker.update(color=list(fills), line={"color": list(edges), "width": 1})
        bar_kw = {
            "x": table.index,
            "y": table[name],
            "name": name,
            "marker": marker,
            "marker_pattern_shape": "/" if name == AVERAGE else "",
            "hovertemplate": "%{x}: <b>%{y:.1f}</b> calls<extra>" + name + "</extra>",
        }
        if single:
            bar_kw["text"] = [f"{v:.0f}" for v in table[name]]
            bar_kw["textposition"] = "outside"
            bar_kw["cliponaxis"] = False
        fig.add_trace(go.Bar(**bar_kw))
    fig = _layout(fig.update_layout(barmode="group"), "Average calls per shift", show_legend=not single)
    if single:
        top = float(table.iloc[:, 0].max()) * 1.15
        fig.update_yaxes(range=[0, max(top, 1.0)])
    return fig


def station_means(history, station_ids):
    """Average calls per shift for every station, in station number order."""
    return history.groupby("Station")["calls"].mean().reindex(station_ids)


def station_bar_chart(means, picked=()):
    """All stations' averages on the mint-to-rose scale; picked stations in their comparison
    colours; the average station as a dashed line."""
    picked = [s for s in picked if s in means.index][:MAX_COMPARE]
    shades = [day_colors(v, means.max()) for v in means]
    # Every bar has a thin darker edge of its colour so the pale ones show on the panel.
    pick_edges = dict(zip(COLORS, STATION_EDGES))
    colors = [station_color(picked, s) if s in picked else f for s, (f, _) in zip(means.index, shades)]
    edges = [pick_edges.get(c, e) for c, (_, e) in zip(colors, shades)]
    fig = go.Figure(go.Bar(
        x=list(means.index), y=means.to_numpy(),
        marker={"color": colors, "cornerradius": 4, "line": {"color": edges, "width": 1}},
        hovertemplate="%{x}: <b>%{y:.1f}</b> calls per shift<extra></extra>", showlegend=False,
    ))
    average = means.mean()
    fig.add_hline(
        y=average, line={"color": AVERAGE_COLOR, "dash": "dash", "width": 1.6},
        annotation_text=f"Average station {average:.1f}", annotation_position="top right",
        annotation_font={"color": INK, "size": FS_XS, "family": FONT_FAMILY},
        annotation_bgcolor=SURFACE, annotation_bordercolor=BORDER_COLOR,
    )
    fig = _layout(fig, "Average calls per shift", height=370, show_legend=False)
    fig.update_layout(margin={"l": 8, "r": 12, "t": 20, "b": 56})
    fig.update_xaxes(tickangle=-90, tickfont={"size": FS_XS})
    return fig


def category_chart(table, counts=None):
    """100% bars: one bar per station, split by category.

    counts (same shape, from category_means) adds the average calls per shift to the hover
    text, so the share and the actual number are both visible.
    """
    fig = go.Figure()
    for category in table.columns:
        hover = "%{y}: <b>%{x:.0f}%</b> " + category.lower()
        extra = {}
        if counts is not None:
            extra["customdata"] = counts[category].to_numpy()
            hover += " (about %{customdata:.1f} calls per shift)"
        fig.add_trace(go.Bar(
            y=table.index, x=table[category], name=category, orientation="h",
            marker={"color": CATEGORY_COLORS[category], "cornerradius": 4,
                    "line": {"color": CATEGORY_EDGES[category], "width": 1}},
            text=[f"{v:.0f}%" if v >= 12 else "" for v in table[category]],
            textposition="inside", insidetextanchor="middle",
            textfont={"size": FS_XS, "family": FONT_FAMILY, "color": CATEGORY_TEXT[category]},
            hovertemplate=hover + "<extra></extra>", **extra,
        ))
    fig = _layout(fig, "", height=170 + 48 * len(table), show_legend=True, legend_top=True)
    fig.update_layout(
        barmode="stack",
        margin={"l": 8, "r": 12, "t": 50, "b": 34},
        xaxis={
            "title": {"text": "Share of calls (%)", "font": AXIS_TITLE_FONT}, "range": [0, 100],
            "showgrid": True, "gridcolor": GRID_COLOR, "griddash": "dot",
        },
        legend={"orientation": "h", "y": 1.24 if len(table) <= 2 else 1.16, "yanchor": "top", "x": 0, "traceorder": "normal"},
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig
