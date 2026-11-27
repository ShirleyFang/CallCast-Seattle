"""Shared chart theme: the app palette, the colours made from it, and chart builders."""

import altair as alt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# The six palette colours. Every other colour below is a lighter or darker step of one
# of them, and style.css and .streamlit/config.toml use these same values.
LIGHT_BLUE = "#99CDD8"
MINT = "#DAEBE3"
CREAM = "#FDE8D3"
SALMON = "#F3C3B2"
SAGE = "#CFD6C4"
SAGE_GREY = "#657166"

# Page, cards and text. Text is sage grey made darker, so it passes 4.5:1 on every surface.
PAGE_BG = "#EDF5F1"  # mint, half way to white
SURFACE = "#FFFBF6"  # cream, nearly white: cards, expanders and the date field
PANEL_BG = "#F1FCFB"  # very light blue: pop-up windows and Explore chart cards
INK = "#263028"
MUTED_INK = "#4F5A50"
# The one type scale, in px. style.css reads it as --fs-* (css_variables below) and the
# charts and maps use the same numbers, so no text anywhere is under 14 px.
FS_XS = 14  # chart ticks, legends, map labels, small sub-labels
FS_SM = 15  # chart titles, hover cards, notes and fact chips
FS_BASE = 16  # captions, buttons, tabs, pickers and the station table
FS_MD = 17  # body text and help lines
FS_LG = 20  # card titles and section heads
FS_XL = 26  # page title
FS_NUM = 38  # card numbers
TYPE_SCALE = {"xs": FS_XS, "sm": FS_SM, "base": FS_BASE, "md": FS_MD, "lg": FS_LG,
              "xl": FS_XL, "num": FS_NUM}
# Axis titles set as plain strings lose the theme font, so builders pass this one.
AXIS_TITLE_FONT = {"color": MUTED_INK, "size": FS_SM}
BORDER_COLOR = SAGE
GRID_COLOR = "#E4E9DE"
HEADER_BG = LIGHT_BLUE  # the flat light blue, now only the key's shaded-days swatch
# The top bar's sky, light at the top left and bluer to the right, from the owner's sky picture.
SKY_TOP = "#E6F4FD"
SKY_MID = "#C6E3F8"
SKY_LOW = "#A8D3F0"  # darkest spot: header ink 8.62:1, captions 7.23:1
HEADER_EDGE = "#323C34"  # sage grey made dark for borders on the sky header
HEADER_MUTED = "#323C34"  # captions on the sky header bar
LIGHT_TEXT = "#FFFAF3"  # on dark hover boxes and dark tags
# Salmon made dark enough for tab labels, buttons and focus rings (5:1 on the page).
ACCENT = "#A04E38"
ACCENT_DARK = "#83402D"
LINK = "#2D6675"  # light blue made darker
FONT_FAMILY = "Figtree, -apple-system, BlinkMacSystemFont, sans-serif"
# Lexend only for the sentences that talk to the reader, like "Saturday looks a little busier
# than usual."; style.css sets the same font on the weather and event alert lines.
MESSAGE_FONT = "Lexend, Figtree, sans-serif"

# Prediction meaning colours: salmon side busier, light-blue side quieter, cream about usual.
# The busier end is darker than the quieter end, so the two sides still differ in grayscale.
CORAL = "#E48A6F"
TERRACOTTA = "#B5573C"
TERRACOTTA_DARK = "#8A3825"
DEEP_BLUE = "#62A0B0"
DEEPER_BLUE = "#22525F"
SAGE_DARK = "#9AA48E"
# Map scale from quieter to busier; grey values 143, 191, 236, 162, 78 (out of 255).
DIVERGE_STOPS = [DEEP_BLUE, LIGHT_BLUE, CREAM, CORAL, TERRACOTTA_DARK]
BUSY_TINT, BUSY_TINT_SOFT, BUSY_TEXT = SALMON, "#FBE6DE", TERRACOTTA_DARK
QUIET_TINT, QUIET_TINT_SOFT, QUIET_TEXT = LIGHT_BLUE, "#E1F0F3", DEEPER_BLUE

# Explore and the history window use the mint-to-rose day scale (DAY_STOPS below); picked
# stations keep their map and card colours there too. Grey stays for the average lines.
CITY_NEUTRAL = SAGE_GREY
AVERAGE_GREY = "#7D877E"
# Picks 1, 2, 3: lavender, warm yellow and mint. Dark text passes on each (6.44, 9.66,
# 10.73:1). Yellow and mint are close in grayscale (grey 212, 220), so tags, cards and table
# rows also carry the pick number. Each pick has a darker edge of the same hue, so its map rim
# and bars still show on pale areas and the light page.
LAVENDER = "#BCA7ED"
YELLOW = "#F8D66F"
PICK_MINT = "#B5F0DD"
STATION_COLORS = [LAVENDER, YELLOW, PICK_MINT]
STATION_EDGES = ["#44357A", "#64500F", "#1F5C49"]
STATION_TEXT = [INK, INK, INK]
# Gold for warnings and clear event effects; moss and gold also mark the two event stations.
MOSS = "#4A6B45"
OCHRE = "#D6AC4E"
OCHRE_TINT, OCHRE_TEXT, OCHRE_EDGE = "#F8ECD3", "#6E4E12", "#96701F"
MOSS_EDGE, MOSS_TINT = "#34502F", "#E3ECDF"
# Weather: muted cold blue and brick, apart from Prediction's quieter and busier steps.
COLD_BLUE, WARM_BRICK = "#5B8E9B", "#B4664D"
# Call types from the day scale's family (grey 99, 153, 203, 231), so stacked bands also read
# in grayscale; Medical, the biggest band, is the softer salmon so the stack is not one dark
# block. Each has a darker edge so the pale ones show on the panel.
CATEGORY_COLORS = {
    "Fire": "#8E4F5A",
    "Medical": "#CF8379",
    "Non-emergency": "#EBC3A1",
    "Other": "#D6EFE9",
}
# Text written on each call type's bar: light on deep rose (5.9:1), dark on the rest.
CATEGORY_TEXT = {"Fire": LIGHT_TEXT, "Medical": INK, "Non-emergency": INK, "Other": INK}

PREDICTION_TONES = {
    "busier": {"main": TERRACOTTA, "dark": TERRACOTTA_DARK, "band": SALMON, "border": CORAL},
    "quieter": {"main": DEEP_BLUE, "dark": DEEPER_BLUE, "band": LIGHT_BLUE, "border": DEEP_BLUE},
    "usual": {"main": SAGE_GREY, "dark": INK, "band": MINT, "border": SAGE_DARK},
}

# Last 4 weeks bars, fewer calls to more: mint, cream, peach, salmon, rose.
DAY_STOPS = ["#CEECE7", "#DCE8E0", "#EAE8D9", "#EFCEB3", "#EAA899", "#DF9FA3"]
# Deep rose for lines drawn over the day scale: 5.9:1 on the panel, 3.3:1 over the rose top.
ROSE_LINE = "#8E4F5A"

PLOT_CONFIG = {"displayModeBar": False}


def rgb(hex_color, alpha=None):
    # "#657166" -> [101, 113, 102], plus alpha when given, for the pydeck maps.
    out = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    return out if alpha is None else out + [alpha]


def hex_to_rgba(hex_color, alpha):
    r, g, b = rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


def mix(a, b, t):
    # The colour t of the way from a to b: mix("#000000", "#FFFFFF", 0.5) is "#808080".
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(rgb(a), rgb(b)))


def ramp(stops, t):
    # The colour at t (0 to 1) along evenly spaced stops.
    t = min(max(t, 0), 1) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return mix(stops[i], stops[i + 1], t - i)


for _tone in PREDICTION_TONES.values():
    _tone["area"] = hex_to_rgba(_tone["main"], 0.14)
CATEGORY_EDGES = {k: mix(c, INK, 0.35) for k, c in CATEGORY_COLORS.items()}


def day_colors(v, top, low=0):
    # A value's fill on the mint-to-rose scale and its edge, a darker step of that fill that
    # gets bolder for bigger values, so pale bars still show on light backgrounds.
    # The scale runs from low (mint, 0 calls unless given) to top (rose).
    t = min(max((v - low) / (top - low), 0), 1) if top > low else 0
    fill = ramp(DAY_STOPS, t)
    return fill, mix(fill, INK, 0.3 + 0.3 * t)



def scale_top(values):
    # The top of a colour scale shared by every station: the biggest known value among them,
    # so a quiet station stays mint and only the busiest reaches rose. Gaps (None) are skipped.
    known = [float(v) for v in values if pd.notna(v)]
    return max(known, default=0) or 1.0

# CSS names for style.css, so the stylesheet uses the same colours as the charts.
CSS_COLORS = {
    "page": PAGE_BG, "surface": SURFACE, "panel": PANEL_BG, "mint": MINT, "salmon": SALMON, "sage": SAGE,
    "ink": INK, "muted": MUTED_INK, "header": HEADER_BG, "header-edge": HEADER_EDGE,
    "header-muted": HEADER_MUTED, "light-text": LIGHT_TEXT, "accent": ACCENT, "accent-dark": ACCENT_DARK,
    "accent-tint": BUSY_TINT_SOFT, "map": MINT, "track": GRID_COLOR,
    "sky-top": SKY_TOP, "sky-mid": SKY_MID, "sky-low": SKY_LOW,
    "sky-wisp": hex_to_rgba(PANEL_BG, 0.6), "sky-haze": hex_to_rgba(PANEL_BG, 0.35),
    **{f"pick-{i}": c for i, c in enumerate(STATION_COLORS)},
    **{f"pick-text-{i}": c for i, c in enumerate(STATION_TEXT)},
    **{f"pick-edge-{i}": c for i, c in enumerate(STATION_EDGES)},
}


def css_variables():
    shadow = hex_to_rgba(INK, 0.06)
    pairs = "".join(f"--cc-{k}: {v}; " for k, v in CSS_COLORS.items())
    pairs += "".join(f"--fs-{k}: {v}px; " for k, v in TYPE_SCALE.items())
    return f":root {{ {pairs}--cc-shadow: {shadow}; }}"


def day_gradient(top, alpha=(0.3, 0.8)):
    # A soft fill from mint at 0 calls up to rose at top, fading in as it climbs.
    n = len(DAY_STOPS) - 1
    scale = [
        [i / n, hex_to_rgba(c, round(alpha[0] + (alpha[1] - alpha[0]) * i / n, 2))]
        for i, c in enumerate(DAY_STOPS)
    ]
    return {"type": "vertical", "colorscale": scale, "start": 0, "stop": top}


def add_gap_fills(fig, series, color, gradient=None):
    # One fill per unbroken run of shifts: a single filled line would bridge the missing ones.
    # A gradient (day_gradient) colours every run on the same scale instead of one flat colour.
    runs = series.isna().cumsum()[series.notna()]
    for _, run in series.dropna().groupby(runs):
        fig.add_trace(go.Scatter(
            x=run.index, y=run.to_numpy(), mode="lines", line={"width": 0}, fill="tozeroy",
            fillcolor=color, fillgradient=gradient, hoverinfo="skip", showlegend=False,
        ))


def add_gap_bridges(fig, series, color, fmt="%a %b %d"):
    # Missing shifts between known ones: a dashed line, the line colour half way to the panel,
    # joins the known neighbours, and an unseen point on each missing day says so on hover.
    # Sep 17 sits between Sep 16 and Sep 18.
    filled = series.interpolate(limit_area="inside")
    missing = series.isna() & filled.notna()
    if not missing.any():
        return
    xs, ys = [], []
    for _, run in filled[missing].groupby((~missing).cumsum()[missing]):
        ends = [series.index.get_loc(run.index[0]) - 1, series.index.get_loc(run.index[-1]) + 1]
        xs += [*series.index[ends], None]
        ys += [*series.iloc[ends], None]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
        line={"color": mix(color, PANEL_BG, 0.5), "width": 1.5, "dash": "dash"},
    ))
    fig.add_trace(go.Scatter(
        x=filled[missing].index, y=filled[missing].to_numpy(), mode="markers",
        marker={"size": 12, "opacity": 0}, name="No data", showlegend=False,
        hovertemplate="%{x|" + fmt + "}: No data for this shift (data feed gap)<extra></extra>",
    ))


def apply_plotly_theme(fig, y_title="", height=320, show_legend=False, legend_top=False):
    # One font, light dotted gridlines and dark hover boxes for every Plotly chart.
    top_margin = 46 if (show_legend and legend_top) else 20
    bottom_margin = 54 if (show_legend and not legend_top) else 32
    # The legend size is set here: Streamlit's chart theme would shrink it to 12 px.
    legend_cfg = {"orientation": "h", "x": 0, "xanchor": "left", "font": {"size": FS_XS}}
    if legend_top:
        legend_cfg.update({"y": 1.18, "yanchor": "top"})
    else:
        legend_cfg.update({"y": -0.22, "yanchor": "top"})
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 12, "t": top_margin, "b": bottom_margin},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": INK, "size": FS_SM},
        showlegend=show_legend,
        legend=legend_cfg,
        hoverlabel={
            "bgcolor": INK,
            "bordercolor": INK,
            "font": {"family": FONT_FAMILY, "color": LIGHT_TEXT, "size": FS_SM},
        },
        xaxis={
            "showgrid": False,
            "linecolor": BORDER_COLOR,
            "linewidth": 1,
            "ticks": "outside",
            "tickcolor": BORDER_COLOR,
            "tickfont": {"color": MUTED_INK, "size": FS_XS},
            "title_font": {"color": MUTED_INK, "size": FS_SM},
            "zeroline": False,
        },
        yaxis={
            "title": y_title,
            "title_font": {"color": MUTED_INK, "size": FS_SM},
            "gridcolor": GRID_COLOR,
            "griddash": "dot",
            "zeroline": False,
            "rangemode": "tozero",
            "tickfont": {"color": MUTED_INK, "size": FS_XS},
        },
    )
    return fig


def history_line_chart(
    series,
    tick_format="%b %d",
    color=ROSE_LINE,
    show_markers=False,
):
    # Line over the mint-to-rose fill, with a dotted average line.
    first, last = series.index.min(), series.index.max()
    peak_val = float(series.max())
    top = max(peak_val * 1.22, 1.0)
    mode = "lines+markers" if show_markers else "lines"
    trace_kw = {
        "x": series.index,
        "y": series.to_numpy(),
        "mode": mode,
        "line": {"color": color, "width": 3},
        "connectgaps": False,
        "hovertemplate": "%{x|" + tick_format + "}: <b>%{y:.1f}</b> calls<extra></extra>",
    }
    if show_markers:
        trace_kw["marker"] = {"size": 4.5, "color": color, "line": {"width": 1, "color": SURFACE}}
    fig = go.Figure()
    # The fill runs straight through missing shifts; the line shows them as dashed bridges.
    add_gap_fills(
        fig, series.interpolate(limit_area="inside"), hex_to_rgba(color, 0.13), day_gradient(top)
    )
    add_gap_bridges(fig, series, color, tick_format)
    fig.add_trace(go.Scatter(**trace_kw))
    mean_val = float(series.mean())
    if series.notna().any():
        fig.add_hline(
            y=mean_val,
            line={"color": MUTED_INK, "dash": "dot", "width": 1.25},
            annotation_text=f"Avg {mean_val:.1f}",
            annotation_position="top left",
            annotation_font={"color": INK, "size": FS_XS, "family": FONT_FAMILY},
            annotation_bgcolor=SURFACE,
            annotation_bordercolor=BORDER_COLOR,
        )
    apply_plotly_theme(fig, y_title="Calls per shift", height=320, show_legend=False)
    # minallowed and maxallowed stop the zoom at the data; without them the line shrinks
    # to a sliver among empty dates and negative call counts.
    fig.update_layout(
        dragmode="pan",
        xaxis={
            "tickformat": tick_format,
            "range": [first, last],
            "minallowed": first,
            "maxallowed": last,
        },
        yaxis={
            "range": [0, top],
            "minallowed": 0,
            "maxallowed": top,
        },
    )
    return fig


def history_window_chart(
    calls, target, usual, expected, low, high, days=28, tone="usual", scale=None
):
    # The station's known shifts on the left (dashed bridges where data is missing, the
    # target's weekday as dots, the usual as a dotted line) and this shift's estimate with its likely range on
    # the right. The right panel has its own axis, so a far-off date does not squash the line.
    target = pd.Timestamp(target)
    last = calls.index.max()
    start = calls.index.min()
    if days is not None:
        start = max(start, last - pd.Timedelta(days=days - 1))
    line = calls.reindex(pd.date_range(start, last, freq="D"))
    same = line[(line.index.dayofweek == target.dayofweek) & line.notna()]
    palette = PREDICTION_TONES.get(tone, PREDICTION_TONES["usual"])
    hover = "%{x|%a %b %d}: <b>%{y:.0f}</b> calls<extra></extra>"
    # Years of daily shifts are a blur, so the long view fades them and adds a 4-week average
    # instead of the weekday dots.
    long_view = days is None or days > 91
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.025
    )
    peak = float(line.max()) if line.notna().any() else 0.0
    top = max(peak, high, usual, 1.0) * 1.15
    # The fill runs straight through missing shifts; the line shows them as dashed bridges.
    # The fill goes mint near 0 to rose at scale (the table's Last 4 weeks top, shared by every
    # station, or this chart's top when not given), paler in the long view.
    add_gap_fills(
        fig, line.interpolate(limit_area="inside"), hex_to_rgba(ROSE_LINE, 0.13),
        day_gradient(scale or top, (0.2, 0.5) if long_view else (0.3, 0.8)),
    )
    add_gap_bridges(fig, line, ROSE_LINE)
    shift_line = {"color": ROSE_LINE, "width": 3}
    if long_view:
        shift_line = {"color": hex_to_rgba(ROSE_LINE, 0.35), "width": 1}
    fig.add_trace(go.Scatter(
        x=line.index, y=line.to_numpy(), mode="lines", name="Calls per shift",
        connectgaps=False, line=shift_line, hovertemplate=hover,
    ), row=1, col=1)
    if long_view:
        avg = line.rolling(28, min_periods=14).mean()
        fig.add_trace(go.Scatter(
            x=avg.index, y=avg.to_numpy(), mode="lines", name="4-week average",
            connectgaps=False, line={"color": ROSE_LINE, "width": 3},
            hovertemplate="%{x|%b %d, %Y}: 4-week average <b>%{y:.1f}</b><extra></extra>",
        ), row=1, col=1)
    else:
        # Zoomed in to 4 weeks, the weekday dots carry their counts, the same as the summary.
        fig.add_trace(go.Scatter(
            x=same.index, y=same.to_numpy(), name=f"{target:%A}s",
            mode="markers+text" if days <= 28 else "markers",
            marker={"size": 8, "color": ROSE_LINE, "line": {"width": 1.5, "color": SURFACE}},
            text=[f"{v:.0f}" for v in same], textposition="top center",
            textfont={"color": ROSE_LINE, "size": FS_SM}, hovertemplate=hover,
        ), row=1, col=1)
    label = f"{target:%a %b %d}"
    fig.add_trace(go.Scatter(
        x=[label], y=[expected], mode="markers+text", name="Expected, likely range",
        marker={"size": 13, "color": palette["main"], "line": {"width": 2, "color": SURFACE}},
        error_y={
            "type": "data", "symmetric": False,
            "array": [high - expected], "arrayminus": [expected - low],
            "color": palette["dark"], "thickness": 2, "width": 9,
        },
        text=[f"{expected:.1f}"], textposition="middle right", cliponaxis=False,
        textfont={"color": INK, "size": FS_SM},
        hovertemplate=(
            f"{label}: expected <b>{expected:.1f}</b><br>likely {low:.0f} to {high:.0f}"
            "<extra></extra>"
        ),
    ), row=1, col=2)
    # Usual for this weekday: a dotted line across both panels, named once in the legend.
    usual_line = {"color": MUTED_INK, "dash": "dot", "width": 1.5}
    fig.add_trace(go.Scatter(
        x=[start, last], y=[usual, usual], mode="lines", line=usual_line,
        name=f"Usual for a {target:%A} ({usual:.1f})", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_hline(y=usual, line=usual_line, row=1, col=2)
    # The legend is HTML above the chart (history_legend_html), so it wraps on a phone.
    apply_plotly_theme(fig, height=320, show_legend=False)
    # No dragging or scroll zoom: the span buttons zoom, and a phone swipe still scrolls.
    fig.update_layout(
        dragmode=False, hovermode="closest", margin={"l": 8, "r": 12, "t": 14, "b": 32}
    )
    fig.update_xaxes(
        fixedrange=True, linecolor=BORDER_COLOR, ticks="outside", tickcolor=BORDER_COLOR,
        tickfont={"color": MUTED_INK, "size": FS_XS},
    )
    half_day = pd.Timedelta(hours=12)
    fig.update_xaxes(
        tickformat="%b %Y" if long_view else "%b %d",
        range=[start - half_day, last + half_day], row=1, col=1,
    )
    fig.update_xaxes(
        type="category", title={"text": "This shift", "font": AXIS_TITLE_FONT}, row=1, col=2
    )
    fig.update_yaxes(
        fixedrange=True, range=[0, top], gridcolor=GRID_COLOR, griddash="dot", zeroline=False,
        tickfont={"color": MUTED_INK, "size": FS_XS},
    )
    fig.update_yaxes(title={"text": "Calls per shift", "font": AXIS_TITLE_FONT}, row=1, col=1)
    return fig


def hourly_altair_chart(hourly, top=None):
    # One bar per shift hour, coloured on the mint-to-rose scale by its calls, the busiest also
    # outlined in ink and labelled, and a dashed average. Rose is top, the busiest hour of any
    # station, so a quiet station stays mint; without top, this chart's own busiest hour.
    df = hourly[["hour", "expected_calls", "share"]].copy()
    df["pct"] = (df["share"] * 100).round(1)
    peak_idx = df["expected_calls"].idxmax()
    peak_hour = df.loc[peak_idx, "hour"]
    peak_val = float(df.loc[peak_idx, "expected_calls"])
    avg_val = float(df["expected_calls"].mean())
    df["fill"], df["edge"] = zip(*(day_colors(v, top or peak_val) for v in df["expected_calls"]))
    df["is_peak"] = df["hour"] == peak_hour
    df.loc[df["is_peak"], "edge"] = INK
    df["edge_width"] = df["is_peak"].map({True: 1.5, False: 0.75})
    order = list(df["hour"])
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "hour:N",
                sort=order,
                title=None,
                axis=alt.Axis(
                    labelAngle=-45,
                    labelColor=MUTED_INK,
                    labelFont=FONT_FAMILY,
                    labelFontSize=FS_XS,
                    tickColor=BORDER_COLOR,
                    domainColor=BORDER_COLOR,
                ),
            ),
            y=alt.Y(
                "expected_calls:Q",
                title="Expected calls",
                scale=alt.Scale(domain=[0, max(peak_val * 1.18, 0.5)]),
                axis=alt.Axis(
                    gridColor=GRID_COLOR,
                    gridDash=[2, 2],
                    domain=False,
                    ticks=False,
                    labelColor=MUTED_INK,
                    titleColor=MUTED_INK,
                    labelFont=FONT_FAMILY,
                    titleFont=FONT_FAMILY,
                    labelFontSize=FS_XS,
                    titleFontSize=FS_SM,
                ),
            ),
            color=alt.Color("fill:N", scale=None, legend=None),
            stroke=alt.Stroke("edge:N", scale=None, legend=None),
            strokeWidth=alt.StrokeWidth("edge_width:Q", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("hour:N", title="Hour"),
                alt.Tooltip("expected_calls:Q", title="Expected calls", format=".2f"),
                alt.Tooltip("pct:Q", title="Usual share (%)", format=".1f"),
            ],
        )
    )
    rule = (
        alt.Chart(df)
        .mark_rule(color=MUTED_INK, strokeDash=[4, 4], strokeWidth=1.2)
        .encode(y=alt.datum(avg_val))
    )
    peak_text = (
        alt.Chart(df[df["is_peak"]])
        .mark_text(
            dy=-9, fontSize=FS_XS, fontWeight=600, font=FONT_FAMILY, color=INK,
            text=f"Peak {peak_val:.1f}",
        )
        .encode(x=alt.X("hour:N", sort=order), y=alt.Y("expected_calls:Q"))
    )
    return (
        (bars + rule + peak_text)
        .properties(height=245, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
    )


def call_mix_chart(mix_df, station_colors=None):
    # Average past calls per shift by call type, one bar colour per type or per station.
    types = list(mix_df.index)
    names = list(mix_df.columns)
    fig = go.Figure()
    if len(names) == 1:
        col = names[0]
        vals = mix_df[col].to_numpy()
        fig.add_trace(
            go.Bar(
                y=types,
                x=vals,
                orientation="h",
                marker={
                    "color": [CATEGORY_COLORS.get(t, CITY_NEUTRAL) for t in types],
                    "cornerradius": 5,
                    "line": {"color": [CATEGORY_EDGES.get(t, INK) for t in types], "width": 1},
                },
                text=[f"{v:.1f}" for v in vals],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: <b>%{x:.1f}</b> calls per shift<extra>" + col + "</extra>",
                showlegend=False,
            )
        )
        apply_plotly_theme(fig, y_title="", height=195, show_legend=False)
        top = float(vals.max()) * 1.18 if len(vals) else 10
        fig.update_layout(
            margin={"l": 8, "r": 38, "t": 10, "b": 28},
            xaxis={
                "title": {"text": "Average calls per shift", "font": AXIS_TITLE_FONT}, "range": [0, max(top, 1.0)],
                "showgrid": True, "gridcolor": GRID_COLOR, "griddash": "dot",
            },
            yaxis={"autorange": "reversed", "showgrid": False},
        )
        return fig

    palette = station_colors or STATION_COLORS
    # A thin darker edge keeps the pale pick bars visible on the light page.
    edges = dict(zip(STATION_COLORS, STATION_EDGES))
    for i, name in enumerate(names):
        vals = mix_df[name].to_numpy()
        fig.add_trace(
            go.Bar(
                y=types,
                x=vals,
                name=name,
                orientation="h",
                marker={"color": (c := palette[i % len(palette)]), "cornerradius": 4,
                        "line": {"color": edges.get(c, c), "width": 1}},
                hovertemplate="%{y}: <b>%{x:.1f}</b> calls per shift<extra>" + name + "</extra>",
            )
        )
    apply_plotly_theme(fig, y_title="", height=245, show_legend=True, legend_top=True)
    fig.update_layout(
        barmode="group",
        margin={"l": 8, "r": 12, "t": 48, "b": 34},
        legend={"orientation": "h", "y": 1.24, "yanchor": "top", "x": 0},
        xaxis={
            "title": {"text": "Average calls per shift", "font": AXIS_TITLE_FONT},
            "showgrid": True, "gridcolor": GRID_COLOR, "griddash": "dot",
        },
        yaxis={"autorange": "reversed", "showgrid": False},
    )
    return fig


def event_impact_chart(effects):
    # Extra calls on event days for each venue and station; clear effects use the venue
    # map's colours (moss green SFD 22, gold SFD 10), the rest the mint-to-rose scale by size.
    df = effects.sort_values("extra_calls")
    # Two venues can share a label and station (stadium concerts near SFD 5), and plotly
    # would stack them on one bar, so those rows also name the venue.
    labels = df["label"] + " (" + df["station"] + ")"
    labels = labels.where(
        ~labels.duplicated(keep=False), df["label"] + " at " + df["venue"] + " (" + df["station"] + ")"
    ).tolist()
    top = float(df["extra_calls"].max())
    bar_colors, edges = [], []
    for r in df.itertuples():
        fill, edge = day_colors(r.extra_calls, top)
        if r.clear:
            fill = edge = MOSS if r.station == "SFD 22" else OCHRE
        bar_colors.append(fill)
        edges.append(edge)
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=df["extra_calls"].to_numpy(),
            orientation="h",
            marker={"color": bar_colors, "cornerradius": 5, "line": {"color": edges, "width": 1}},
            text=[f"{r.extra_calls:+.1f} ({r.extra_pct:+.0f}%)" for r in df.itertuples()],
            textposition="outside",
            cliponaxis=False,
            customdata=list(zip(df["venue"], df["events"], df["usual_calls"], df["event_calls"])),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{customdata[0]} · %{customdata[1]} events<br>"
                "Usual %{customdata[2]:.1f} → Event day %{customdata[3]:.1f}<br>"
                "Change: <b>%{x:+.1f}</b> calls<extra></extra>"
            ),
            showlegend=False,
        )
    )
    apply_plotly_theme(fig, y_title="", height=110 + 34 * len(df), show_legend=False)
    min_x = min(float(df["extra_calls"].min()), 0.0)
    max_x = max(float(df["extra_calls"].max()), 1.0)
    pad_l = 1.6 if min_x < 0 else 0.3
    fig.add_vline(x=0, line={"color": BORDER_COLOR, "width": 1.5})
    fig.update_layout(
        margin={"l": 10, "r": 76, "t": 12, "b": 32},
        xaxis={
            "title": {"text": "Extra calls per event day", "font": AXIS_TITLE_FONT},
            "range": [min_x - pad_l, max_x * 1.28],
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "griddash": "dot",
        },
        yaxis={"showgrid": False},
    )
    return fig


def style_event_table(table, clear=True):
    look = f"color: {MUTED_INK};"
    if clear:
        look = f"background-color: {OCHRE_TINT}; color: {OCHRE_TEXT}; font-weight: 600;"
    return table.style.map(lambda _: look, subset=["Extra calls", "Extra %"])
