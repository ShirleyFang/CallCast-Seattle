"""HTML builders for the Prediction tab: the briefing callout, range bars, map legend, the
history window's legend and the clickable station table."""

from html import escape

import pandas as pd

from chart_theme import (
    BORDER_COLOR,
    BUSY_TEXT,
    BUSY_TINT,
    BUSY_TINT_SOFT,
    DAY_STOPS,
    DIVERGE_STOPS,
    GRID_COLOR,
    INK,
    MESSAGE_FONT,
    MUTED_INK,
    PREDICTION_TONES,
    QUIET_TEXT,
    QUIET_TINT,
    QUIET_TINT_SOFT,
    ROSE_LINE,
    SAGE_GREY,
    STATION_COLORS,
    STATION_TEXT,
    SURFACE,
    day_colors,
    hex_to_rgba,
)

SHADOW = f"0 1px 2px {hex_to_rgba(INK, 0.05)},0 4px 12px {hex_to_rgba(INK, 0.04)}"


def briefing_html(sentence, tone="usual"):
    accent = PREDICTION_TONES.get(tone, PREDICTION_TONES["usual"])["main"]
    return (
        f'<div style="background:{SURFACE};border:1px solid {BORDER_COLOR};'
        f'border-left:5px solid {accent};border-radius:10px;padding:14px 18px;'
        f'color:{INK};font-family:{MESSAGE_FONT};font-size:var(--fs-lg);line-height:1.4;'
        'font-weight:500;'
        f'box-shadow:{SHADOW};">'
        f"{escape(sentence)}</div>"
    )


def range_bar_html(spec):
    # Salmon busier, light blue quieter; the dot, tick and band shapes still read in grayscale.
    palette = PREDICTION_TONES[spec["tone"]]

    def num(v):
        return f'<strong style="color:{INK};font-size:var(--fs-base);font-weight:700;">{v}</strong>'

    return (
        '<div style="margin:8px 0 2px 0;font-variant-numeric:tabular-nums;">'
        # Each muted label sits over its bold number, so the three fit a narrow card.
        '<div style="display:flex;justify-content:space-between;gap:4px;flex-wrap:nowrap;'
        f'white-space:nowrap;font-size:var(--fs-xs);font-weight:500;color:{MUTED_INK};'
        'line-height:1.25;margin-bottom:6px;">'
        f'<span style="display:flex;flex-direction:column;">Low{num(spec["low"])}</span>'
        '<span style="display:flex;flex-direction:column;align-items:center;">'
        f'Usual{num(round(spec["usual"], 1))}</span>'
        '<span style="display:flex;flex-direction:column;align-items:flex-end;">'
        f'High{num(spec["high"])}</span>'
        "</div>"
        '<div style="padding:0 8px;">'
        '<div style="position:relative;height:30px;">'
        '<div style="position:absolute;top:11px;left:0;right:0;height:8px;border-radius:999px;'
        f'background:{GRID_COLOR};box-shadow:inset 0 1px 2px {hex_to_rgba(INK, 0.08)};"></div>'
        f'<div style="position:absolute;top:7px;left:{spec["band_left"]:.2f}%;'
        f'width:{spec["band_width"]:.2f}%;height:16px;border-radius:6px;'
        f'background:{palette["band"]};border:1.5px solid {palette["border"]};'
        'box-sizing:border-box;"></div>'
        f'<div style="position:absolute;top:3px;left:{spec["usual_pct"]:.2f}%;'
        f'transform:translateX(-50%);width:3px;height:24px;background:{INK};'
        'border-radius:2px;"></div>'
        f'<div style="position:absolute;top:8px;left:{spec["expected_pct"]:.2f}%;'
        f'transform:translateX(-50%);width:14px;height:14px;border-radius:50%;'
        f'background:{palette["main"]};border:2.5px solid {SURFACE};'
        f'box-shadow:0 0 0 1.5px {INK},0 1px 3px {hex_to_rgba(INK, 0.22)};box-sizing:border-box;"></div>'
        "</div></div>"
        f'<div style="display:flex;flex-wrap:wrap;gap:6px 14px;font-size:var(--fs-xs);font-weight:500;'
        f'color:{MUTED_INK};margin-top:6px;align-items:center;">'
        '<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{palette["main"]};'
        f'border:1.5px solid {INK};display:inline-block;"></span>Expected</span>'
        '<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:3px;height:12px;background:{INK};border-radius:1px;display:inline-block;">'
        "</span>Usual</span>"
        '<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:14px;height:9px;border-radius:3px;background:{palette["band"]};'
        f'border:1.5px solid {palette["border"]};display:inline-block;"></span>Likely range</span>'
        "</div></div>"
    )


def map_legend_html():
    swatches = "".join(
        f'<div style="flex:1;background:{stop};"></div>' for stop in DIVERGE_STOPS
    )
    return (
        '<div style="display:flex;height:11px;border-radius:999px;overflow:hidden;'
        f'border:1px solid {BORDER_COLOR};">{swatches}</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:var(--fs-sm);color:{INK};'
        'font-weight:600;margin-top:6px;">'
        "<span>Quieter than usual</span><span>About usual</span><span>Busier than usual</span></div>"
        f'<div style="font-size:var(--fs-sm);color:{MUTED_INK};margin-top:6px;line-height:1.45;">'
        "Numbers show station and expected calls where space fits. "
        "Picked areas have coloured rims that match their cards.</div>"
    )


def history_legend_html(target, usual, tone="usual", long_view=False):
    # Keys for the history window's chart, matching its marks; HTML so it wraps on a phone.
    palette = PREDICTION_TONES.get(tone, PREDICTION_TONES["usual"])
    day = f"{pd.Timestamp(target):%A}"
    line = f'<span style="width:18px;height:0;border-top:3px solid {ROSE_LINE};"></span>'
    dot = f'<span style="width:8px;height:8px;border-radius:50%;background:{ROSE_LINE};"></span>'
    avg = f'<span style="width:18px;height:0;border-top:3px solid {ROSE_LINE};"></span>'
    usual_mark = f'<span style="width:18px;height:0;border-top:2px dotted {MUTED_INK};"></span>'
    est = (
        f'<span style="width:10px;height:10px;border-radius:50%;background:{palette["main"]};'
        f'box-shadow:0 0 0 1.5px {palette["dark"]};"></span>'
    )
    items = [(line, "Calls per shift")]
    items.append((avg, "4-week average") if long_view else (dot, f"{day}s"))
    items.append((usual_mark, f"Usual for a {day} ({usual:.1f})"))
    items.append((est, "Expected, with its likely range"))
    spans = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;">{mark}{escape(text)}</span>'
        for mark, text in items
    )
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:6px 18px;font-size:var(--fs-sm);font-weight:500;color:{INK};'
        f'margin:2px 0 -4px 0;">{spans}</div>'
    )


# The station table is plain HTML in a small Streamlit component: st.dataframe draws on a
# canvas, so it cannot give one column a hand pointer or shade the row under the mouse.
# A click sends the row's own station name to Python, never a row number.
TABLE_CSS = """
.wrap { overflow-x: auto; border: 1px solid var(--cc-sage); border-radius: 10px;
  background: var(--cc-surface); }
table { width: 100%; min-width: 560px; border-collapse: collapse; font-size: var(--fs-base);
  color: var(--cc-ink); font-variant-numeric: tabular-nums; }
th { text-align: left; font-weight: 600; color: var(--cc-muted); padding: 8px 12px;
  border-bottom: 1px solid var(--cc-sage); white-space: nowrap; vertical-align: bottom; }
th small { display: block; font-weight: 500; font-size: var(--fs-xs); }
th[title] { cursor: help; }
td { padding: 6px 12px; height: 48px; border-bottom: 1px solid var(--cc-track);
  white-space: nowrap; }
/* Last 4 weeks gets room for 28 bars about 20 px wide. On a narrow screen it takes what is
   left; on a wide one the other columns share the extra, so no strip stays empty. */
th:last-child, td:last-child { width: 750px; }
tr:last-child td { border-bottom: 0; }
.go { cursor: pointer; transition: transform var(--cc-hover-time) ease-out; }
tr:has(.go:hover) td { background: var(--cc-mint); }
td.go:hover { box-shadow: inset 0 0 0 2px var(--line); }
.go:focus-visible { outline: 2px solid var(--cc-accent); outline-offset: -2px; }
/* A mouse over a clickable cell lifts it 1 px and slides a 3 px accent bar down the row's
   left edge; a press (any pointer or key) shrinks the cell for a blink. Motion values are the
   --cc-* settings at the top of style.css. */
td:first-child { position: relative; }
td:first-child::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
  background: var(--cc-accent); transform: scaleY(0);
  transition: transform var(--cc-hover-time) ease-out; }
@media (hover: hover) {
  td.go:hover { transform: translateY(var(--cc-lift)); }
  tr:has(.go:hover) td:first-child::before { transform: scaleY(1); }
}
.go.press { animation: press var(--cc-press-time) ease-out; }
@keyframes press { 50% { transform: scale(var(--cc-press)); } }
@media (prefers-reduced-motion: reduce) {
  .go, td:first-child::before { transition: none; }
  td.go:hover { transform: none; }
  .go.press { animation: none; }
}
.box { display: inline-block; width: 18px; height: 18px; border: 1.5px solid var(--line);
  border-radius: 4px; vertical-align: middle; box-sizing: border-box; }
tr.on .box { background: var(--cc-ink); border-color: var(--cc-ink); }
tr.on .box::after { content: "\\2713"; display: block; color: var(--cc-surface);
  font-size: var(--fs-xs); line-height: 15px; text-align: center; }
.track { display: inline-block; width: 90px; height: 8px; border-radius: 999px;
  background: var(--cc-track); vertical-align: middle; margin-right: 8px; overflow: hidden; }
/* The Expected bar shows the mint-to-rose scale across the whole track, so a short bar stays
   mint and only a long one reaches rose. A thin dark edge keeps the mint end off the track. */
.fill { display: block; height: 100%; border-radius: 999px; background: var(--day-scale);
  background-size: 90px 100%; box-shadow: inset 0 0 0 0.5px var(--line); }
/* The 28 bars fill the cell evenly: 5 px wide with 2 px gaps on a phone, up to 20 and 7. */
.bars { display: flex; align-items: flex-end; column-gap: clamp(2px, 1%, 7px); height: 42px;
  width: 100%; min-width: 194px; }
/* One solid shade per day with a thin edge. An estimate keeps its shade, drawn a little lighter. */
.bar { --b: var(--bar); --e: var(--line); flex: 1 1 0; min-width: 5px;
  background: var(--b); border-radius: 2px 2px 0 0; box-shadow: inset 0 0 0 0.5px var(--e); }
.bar.est { opacity: 0.7; }
.ramp { display: inline-block; width: 34px; height: 6px; border-radius: 3px; margin: 0 4px;
  vertical-align: middle; box-shadow: inset 0 0 0 0.5px var(--line); }
.bar.gap { height: 2px; background: var(--cc-track); border-radius: 0; }
"""

TABLE_JS = """
export default function ({ data, parentElement, setTriggerValue }) {
  const wrap = parentElement.querySelector(".wrap");
  wrap.innerHTML = data.html;
  const send = (cell) => {
    const row = cell.closest("tr");
    // Tick at once, but only as Python will: a 4th station is refused and stays unticked.
    const room = wrap.querySelectorAll("tr.on").length < data.limit;
    if (cell.dataset.act === "tick" && (room || row.classList.contains("on"))) {
      row.classList.toggle("on");
    }
    setTriggerValue("action", { act: cell.dataset.act, station: row.dataset.station });
  };
  // The press starts on pointer down, so it plays before the click reruns the page.
  const press = (cell) => {
    cell.classList.remove("press");
    void cell.offsetWidth;
    cell.classList.add("press");
  };
  wrap.onpointerdown = (e) => {
    const cell = e.target.closest("[data-act]");
    if (cell && e.button === 0) press(cell);
  };
  wrap.onanimationend = (e) => e.target.classList.remove("press");
  wrap.onclick = (e) => {
    const cell = e.target.closest("[data-act]");
    if (cell) send(cell);
  };
  wrap.onkeydown = (e) => {
    const cell = e.target.closest("[data-act]");
    if (cell && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      press(cell);
      send(cell);
    }
  };
}
"""

def _diff_cell(val):
    # ▲ busier in salmon, ▼ quieter in light blue, with the 2-call and 0.5-call cut-offs of the
    # map's hover words. Anything that rounds to 0.0 has no arrow.
    if round(val, 1) == 0:
        return f'<td style="color:{MUTED_INK}">0.0</td>'
    tint, ink = (BUSY_TINT_SOFT, BUSY_TEXT) if val > 0 else (QUIET_TINT_SOFT, QUIET_TEXT)
    if abs(val) >= 2:
        tint = BUSY_TINT if val > 0 else QUIET_TINT
    elif abs(val) <= 0.5:
        tint, ink = "transparent", MUTED_INK
    weight = 600 if abs(val) >= 2 else 400
    return (
        f'<td style="background:{tint};color:{ink};font-weight:{weight}">'
        f"{'▲' if val > 0 else '▼'} {val:+.1f}</td>"
    )



def week_bars_html(values, dates, estimated, top, weekday):
    # One bar per day on a scale shared by every row: height and colour both grow with calls.
    # Estimates are drawn a little lighter and gaps are flat. Days on the shift's weekday (0 = Monday)
    # say so in the tooltip: "Sat Oct 17, same weekday: 41 calls".
    bars = []
    for v, d, est in zip(values, dates, estimated):
        day = f"{d:%a %b %d}" + (", same weekday" if d.dayofweek == weekday else "")
        if v is None:
            bars.append(f'<span class="bar gap" title="{day}: no data"></span>')
            continue
        tip = f"about {v:.1f}, an estimate" if est else f"{v:.0f} calls"
        fill, edge = day_colors(v, top)
        bars.append(
            f'<span class="bar{" est" if est else ""}" style="height:{max(v / top * 100, 6):.0f}%;'
            f'--b:{fill};--e:{edge}" title="{day}: {tip}"></span>'
        )
    return f'<span class="bars">{"".join(bars)}</span>'


def ranked_table_html(table, picked, dates, last, target):
    # The busiest-first station table. Clicking Last 4 weeks opens the history window and
    # clicking Expected the hourly breakdown; the tick box picks the station to compare.
    estimated = [d > pd.Timestamp(last) for d in dates]
    top = max((v for w in table["Last 4 weeks"] for v in w if v is not None), default=1) or 1
    most = float(table["Expected calls"].max()) or 1
    target = pd.Timestamp(target)
    sub = f"{dates[0]:%b %d} to {dates[-1]:%b %d}"
    weeks_tip = (
        f"Calls per shift, one bar per day from {sub}, oldest on the left. Point at a bar for "
        f"its date and calls; {target:%A}s, the same weekday as this shift, say so. Every row "
        "uses one scale, so bar heights compare. Lighter = fewer calls, rose = more."
    )
    if any(estimated):
        weeks_tip += (
            f" Calls are known up to {pd.Timestamp(last):%b %d}; later days are estimates, "
            "drawn a little lighter, not real calls."
        )
        sub += f", estimated after {pd.Timestamp(last):%b %d}"
    head = (
        '<tr><th title="Tick to compare"></th><th>Station</th>'
        '<th title="Expected calls for this shift; the bar runs from 0 to the busiest station. '
        'Click a station\'s bar for its hourly breakdown.">Expected<small>click for hours</small></th>'
        '<th title="The likely range for this station, from Low to High.">Likely range</th>'
        '<th title="Expected minus the usual for this weekday: ▲ busier, ▼ quieter.">vs usual</th>'
        f'<th title="{escape(weeks_tip)} Click a station\'s bars for its history.">'
        f"Last 4 weeks<small>{sub}</small>"
        f'<small>fewer<span class="ramp" style="background:linear-gradient(90deg,{",".join(DAY_STOPS)})">'
        "</span>more calls</small></th></tr>"
    )
    rows = []
    for _, r in table.iterrows():
        name = r["Station"]
        on = name in picked
        tint, label = "", escape(name)
        if on:
            i = picked.index(name)
            tint = f' style="background:{STATION_COLORS[i]};color:{STATION_TEXT[i]};font-weight:700"'
            label = f"{i + 1}&nbsp;&nbsp;{escape(name)}"
        rows.append(
            f'<tr data-station="{escape(name)}"{" class=on" if on else ""}>'
            f'<td class="go" data-act="tick" role="checkbox" aria-checked="{str(on).lower()}" '
            f'tabindex="0" title="Tick to compare {escape(name)}"><span class="box"></span></td>'
            f"<td{tint}>{label}</td>"
            f'<td class="go" data-act="hourly" role="button" tabindex="0" '
            f'title="Open {escape(name)}\'s hourly breakdown">'
            f'<span class="track"><span class="fill" style="width:{r["Expected calls"] / most * 100:.1f}%">'
            f'</span></span>{r["Expected calls"]:.1f}</td>'
            f"<td>{escape(r['Likely range'])}</td>"
            f"{_diff_cell(r['Difference'])}"
            f'<td class="go" data-act="history" role="button" tabindex="0" '
            f'title="Open {escape(name)}\'s history">'
            f"{week_bars_html(r['Last 4 weeks'], dates, estimated, top, target.dayofweek)}</td></tr>"
        )
    colours = (
        f"--day-scale:linear-gradient(90deg,{','.join(DAY_STOPS)});"
        f"--bar:{PREDICTION_TONES['usual']['border']};"
        f"--line:{SAGE_GREY}"
    )
    return f'<table style="{colours}"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'
