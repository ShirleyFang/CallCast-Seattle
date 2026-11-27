"""HTML for the app's explanation text, styled by the cc- classes in style.css.

Each pattern takes plain strings or tuples and returns HTML for st.html. Inside a text,
**word** is bold, a markdown link becomes a link chip, and a mark like {tick} draws a picture of
the control or chart mark the sentence talks about (GLYPHS below; {esc} shows the Esc key).
"""

import re
from html import escape, unescape

# Small drawings of the real controls and chart marks, drawn by style.css (st.html drops SVG).
GLYPHS = {
    "tick": "", "bars": "<i></i><i></i><i></i>", "track": "<i></i>", "download": "",
    "point": "", "area": "<i></i>", "dash": "", "shade": "",
}


def glyph(name):
    if name == "esc":
        return '<kbd class="cc-key">Esc</kbd>'
    return f'<span class="cc-glyph cc-{name}" aria-hidden="true">{GLYPHS[name]}</span>'


def _text(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(s, quote=False))
    s = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a class="cc-chip" href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        s,
    )
    return re.sub(r"\{(\w+)\}", lambda m: glyph(m[1]), s)


def lead(text):
    # The main sentence under a heading or chart.
    return f'<p class="cc-lead">{_text(text)}</p>'


def terms(items):
    # Labelled definitions: [("Usual", "The average of earlier shifts on this weekday."), ...]
    rows = "".join(f"<dt>{_text(k)}</dt><dd>{_text(v)}</dd>" for k, v in items)
    return f'<dl class="cc-terms">{rows}</dl>'


def note(text, under=False):
    # A quiet side note with a small info mark in front; the span keeps a bold label on the same line.
    # under=True tucks it right below the message box it explains (the weather alert note).
    return f'<p class="cc-note{" cc-under" if under else ""}"><span>{_text(text)}</span></p>'


def facts(items):
    # Small labelled facts: [("Data through", "**Thu Oct 22 08:00**, the last shift ...")].
    return '<div class="cc-facts">' + "".join(
        f'<span class="cc-fact"><span class="cc-fact-label">{escape(k)}</span> {_text(v)}</span>'
        for k, v in items
    ) + "</div>"


def chips(links):
    # Outside links as small chips: [("Live incidents", "SFD Live", "https://sfdlive.com/"), ...]
    return '<div class="cc-chips">' + "".join(
        f'<span class="cc-chip-row">{escape(label)} <a class="cc-chip" href="{url}" '
        f'target="_blank" rel="noopener noreferrer">{escape(name)}</a></span>'
        for label, name, url in links
    ) + "</div>"


def hints(items):
    # How-to hints in one row, each led by a glyph mark: ["{tick} Tick a row to compare ...", ...]
    return '<div class="cc-hints">' + "".join(
        f'<span class="cc-hint">{_text(t)}</span>' for t in items
    ) + "</div>"


def more(text, title="How to read"):
    # Longer help, closed until clicked; a list of texts gives one paragraph each.
    body = "".join(f"<p>{_text(t)}</p>" for t in ([text] if isinstance(text, str) else text))
    return f'<details class="cc-more"><summary>{escape(title)}</summary>{body}</details>'


def row(*parts):
    return f'<div class="cc-row">{"".join(parts)}</div>'


def block(*parts, card=False, title=None):
    head = f'<p class="cc-title">{escape(title)}</p>' if title else ""
    return f'<div class="cc-text{" cc-card" if card else ""}">{head}{"".join(parts)}</div>'


def panels(*cards):
    # Cards side by side on wide screens, one per line on a phone.
    return f'<div class="cc-panels">{"".join(cards)}</div>'


def plain(html):
    # The words a block shows, as one line: "Likely range: about 4 in 5 ...". Bold and links
    # sit inside a sentence, so they add no space ("**Oct 22**, the" stays "Oct 22, the").
    html = re.sub(r"</?(b|a)\b[^>]*>", "", html)
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).split())
