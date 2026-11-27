"""Soft UI sounds: a tick when the mouse enters something clickable, a click when it is
pressed, and the click out when a pop-up window is closed by its X, Esc or a click outside."""

import streamlit as st

from app_data import ROOT

# The files sit in static/sounds; Streamlit serves that folder at app/static/
# (enableStaticServing in .streamlit/config.toml). Volumes stay low so the sounds never get loud.
# close.wav is the studio's click out (ui_back_dismiss.wav), the sound of every way out.
SOUND_DIR = ROOT / "static" / "sounds"
SOUNDS = {"hover": 0.25, "click": 0.45, "close": 0.45}
SOUNDS_KEY = "ui_sounds"

# What ticks and clicks: buttons (Download, Citywide hourly breakdown, the span and window
# buttons), tabs, links drawn as buttons and the station table's tick, Expected and bars cells.
# Map areas are handled apart, while the hand cursor shows.
CLICKS = ["button", "a[data-testid*='LinkButton']", "[role='tab']", "[data-act]"]
# Boxes and lists stay silent even where they hold a button: the date box and its calendar
# (stDateInputCalendar), the compare list and its names (stMultiSelectDropdown), select boxes,
# any open list, text fields and sliders.
QUIET = [
    "[data-testid^='stDateInput']", "[data-testid^='stMultiSelect']",
    "[data-testid^='stSelectbox']", "[data-testid='stSlider']",
    "[role='listbox']", "input", "textarea",
]

# One listener for the whole page, attached once even though Streamlit reruns this script.
# Nothing plays before the first click: browsers block sound until then, and that click
# turns it on (and plays). Any failure (blocked or missing sound) stays silent.
SOUNDS_JS = """
export default function ({ data }) {
  if (window.ccSounds) return;
  window.ccSounds = true;
  const raw = {}, dec = {};
  let ctx = null, last = null, mapKey = "", x = -1, y = -1, out = 0, open = false;
  for (const snd in data.volume) {
    fetch(new URL(data.url + snd + ".wav", document.baseURI))
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((b) => { raw[snd] = b; })
      .catch(() => {});
  }
  // Called only inside a press the browser counts as a user action, so it never complains.
  const unlock = () => {
    if (!ctx) try { ctx = new AudioContext(); } catch {}
    else if (ctx.state === "suspended") ctx.resume().catch(() => {});
  };
  // Each sound is decoded on its first use, so the click that turns sound on plays too.
  const play = (snd) => {
    if (!ctx) return;
    if (!dec[snd]) {
      if (!raw[snd]) return;
      dec[snd] = ctx.decodeAudioData(raw[snd]);
    }
    dec[snd].then((buf) => {
      const src = ctx.createBufferSource(), gain = ctx.createGain();
      gain.gain.value = data.volume[snd];
      src.buffer = buf;
      src.connect(gain).connect(ctx.destination);
      src.start();
    }, () => {});
  };
  const clicks = data.clicks.join(", "), quiet = data.quiet.join(", ");
  // The clickable thing under the mouse (a button, not the text inside it), or nothing when
  // the mouse is on a box or list, or on a disabled button.
  const unitOf = (e) => {
    for (const el of e.composedPath()) {
      if (!(el instanceof Element)) continue;
      if (el.matches(quiet)) return null;
      if (el.matches(clicks)) {
        return el.matches(":disabled, [aria-disabled='true']") || el.closest(quiet) ? null : el;
      }
    }
    return null;
  };
  const mapOf = (el) => el && el.closest(".stDeckGlJsonChart, [data-testid='stDeckGlJsonChart']");
  // A map area counts as clickable while the hand cursor shows over it (style.css); its hover
  // card names the area. deck.gl draws both a frame after the mouse moves, so look a moment later.
  const checkMap = (map) => setTimeout(() => {
    const tip = map.querySelector(".deck-tooltip"), canvas = map.querySelector("canvas");
    const hand = canvas && getComputedStyle(canvas).cursor === "pointer";
    const key = hand && tip && tip.style.display === "block" ? tip.textContent : "";
    if (key && key !== mapKey) play("hover");
    mapKey = key;
  }, 60);
  document.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    // Moved for real: a page redraw under a still mouse plays nothing.
    const moved = e.screenX !== x || e.screenY !== y;
    x = e.screenX;
    y = e.screenY;
    const map = mapOf(e.composedPath()[0] instanceof Element ? e.composedPath()[0] : null);
    if (map) {
      last = map;
      if (moved) checkMap(map);
      return;
    }
    mapKey = "";
    const unit = unitOf(e);
    if (unit && unit !== last && moved) play("hover");
    last = unit;
  }, true);
  const dialog = () => document.querySelector('[data-testid="stDialog"]');
  document.addEventListener("pointerdown", (e) => {
    // A finger tap counts as a user action only when it lifts (pointerup below).
    if (e.pointerType === "mouse") unlock();
    if (e.button !== 0) return;
    const el = e.composedPath()[0];
    // The window's X or a press outside it is a way out: the click out plays, never a click.
    const box = dialog() && document.querySelector('[role="dialog"]');
    const shut = el instanceof Element && el.closest('button[aria-label="Close"]');
    out = box && (shut || !e.composedPath().includes(box)) ? Date.now() : 0;
    if (out) return;
    const map = mapOf(el instanceof Element ? el : null);
    if (map ? mapKey : unitOf(e)) play("click");
  }, true);
  document.addEventListener("pointerup", (e) => { if (e.pointerType !== "mouse") unlock(); }, true);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") unlock();
    else if (dialog()) out = Date.now();
  }, true);
  // The click out plays once the window has really closed after a way out. A window closed by
  // one of its own buttons plays only that button's click.
  new MutationObserver(() => {
    const now = !!dialog();
    if (open && !now) {
      setTimeout(() => {
        if (!dialog() && Date.now() - out < 1500) play("close");
        out = 0;
      }, 120);
    }
    open = now;
  }).observe(document.body, { childList: true, subtree: true });
}
"""


def add_ui_sounds():
    # Hidden by style.css (.st-key-ui_sounds), so it takes no room on the page.
    sounds = st.components.v2.component(SOUNDS_KEY, html="<span></span>", js=SOUNDS_JS)
    sounds(
        data={"url": "app/static/sounds/", "volume": SOUNDS, "clicks": CLICKS, "quiet": QUIET},
        key=SOUNDS_KEY,
    )
