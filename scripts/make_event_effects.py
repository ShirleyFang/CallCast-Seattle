# What it does
# Saves how calls changed at the stations next to each venue on past event shifts, for the
# app's Explore > Events tab (and for anyone else who needs the same numbers, e.g. a warning):
#   data/event_effects.csv  one row per event kind x venue x nearby station: number of events,
#                           usual calls, calls on event shifts, extra calls, extra %, t,
#                           whether the effect is clear (the Step 14 rule), and the venue's
#                           position
#   data/event_shifts.csv   one row per event shift: calls that shift, the usual calls, and the
#                           difference, for the per-event chart
#
# Same method as scripts/evaluate_events.py (feature_sets.choose_event_pairs): the 2 stations
# nearest each venue; usual = the same station and weekday +-4 weeks without that kind of
# event (at least 3 such shifts); 2022-2024 only (CHOOSE_END), the years the pairs were
# chosen on. A pair is clear with at least 10 events, +10% or more and t >= 2.
#
# Run it from the repository root: uv run python scripts/make_event_effects.py

import numpy as np
import pandas as pd

from cv_utils import load_dataset
from feature_sets import CHOOSE_END, VENUES, calls_by_station, choose_event_pairs, load_events, nearest_stations

EFFECTS_PATH = "data/event_effects.csv"
SHIFTS_PATH = "data/event_shifts.csv"
FIRST_YEAR = 2022
KIND_NAMES = {"stadium concert": "Stadium concert", "arena concert": "Arena concert"}


def event_shift_rows(calls, kind, venue, station, days, before=CHOOSE_END):
    """Each event shift at one station: its calls, the usual calls and the difference.

    The same shifts and numbers as feature_sets.event_effect, one row per event.
    """
    s = calls[station][calls.index < before].dropna()
    days = set(days)
    rows = []
    for d0 in sorted(days & set(s.index)):
        around = [d0 + pd.Timedelta(weeks=k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        around = [d for d in around if d in s.index and d not in days]
        if len(around) >= 3:
            usual = s[around].mean()
            rows.append({"kind": kind, "venue": venue, "station": station, "ShiftDate": d0.date(),
                         "calls": int(s[d0]), "usual_calls": round(usual, 2),
                         "difference": round(s[d0] - usual, 2)})
    return rows


def main():
    df = load_dataset()
    df = df[df["ShiftDate"] < CHOOSE_END]  # only the years the pairs were chosen on
    events = load_events()
    effects, chosen = choose_event_pairs(df, events)
    effects = effects[effects["events"] >= 2].copy()  # one-off events have no effect to show
    effects["event_calls"] = effects["usual_calls"] + effects["extra_calls"]
    effects["clear"] = [(k, v, s) in chosen for k, v, s in zip(effects["kind"], effects["venue"], effects["station"])]
    effects["label"] = effects["kind"].map(KIND_NAMES).fillna(effects["kind"])
    effects["years"] = f"{FIRST_YEAR}-{CHOOSE_END.year - 1}"
    # venue position, for the venue map in the app
    effects["venue_lat"] = effects["venue"].map(VENUES["lat"])
    effects["venue_lon"] = effects["venue"].map(VENUES["lon"])
    effects = effects.sort_values(["clear", "extra_pct"], ascending=[False, False])
    cols = ["label", "kind", "venue", "station", "events", "usual_calls", "event_calls",
            "extra_calls", "extra_pct", "t", "clear", "years", "venue_lat", "venue_lon"]
    effects[cols].round({"usual_calls": 2, "event_calls": 2, "extra_calls": 2, "extra_pct": 1, "t": 2}) \
        .to_csv(EFFECTS_PATH, index=False)

    calls = calls_by_station(df)
    rows = []
    for row in effects.itertuples():
        days = events.loc[(events["kind"] == row.kind) & (events["venue"] == row.venue), "ShiftDate"]
        rows += event_shift_rows(calls, row.kind, row.venue, row.station, days)
    shifts = pd.DataFrame(rows)
    shifts.to_csv(SHIFTS_PATH, index=False)

    # check: the per-event rows give back the same summary as choose_event_pairs
    check = shifts.groupby(["kind", "venue", "station"])["difference"].agg(["size", "mean"])
    merged = effects.set_index(["kind", "venue", "station"]).join(check)
    assert (merged["size"] == merged["events"]).all()
    assert np.allclose(merged["mean"], merged["extra_calls"], atol=0.01)

    print(f"saved {EFFECTS_PATH}: {len(effects)} event-station pairs ({effects['clear'].sum()} clear), "
          f"and {SHIFTS_PATH}: {len(shifts)} event shifts, {FIRST_YEAR}-{CHOOSE_END.year - 1}")
    print(effects[["label", "venue", "station", "events", "usual_calls", "event_calls", "extra_calls",
                   "extra_pct", "t", "clear"]].round(2).to_string(index=False))
    print("venues:", ", ".join(f"{v} -> {', '.join(nearest_stations(v))}" for v in VENUES.index))


if __name__ == "__main__":
    main()
