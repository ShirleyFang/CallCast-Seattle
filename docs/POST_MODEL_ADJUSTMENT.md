# Post-model adjustment for large events

Large events at a stadium raise calls at the fire station next to the venue. The GBM cannot learn this well, because these shifts are rare. So we keep the model as it is and add the extra calls **after** the model predicts.

```
final forecast = GBM forecast + event adjustment
```

The adjustment is 0 for every station-shift that is not on the list below. The model, its features and its training do not change.

Status: proposed, not yet in `predict.py`. The team decides before the final model.

---

## 1. Why not a feature

We tested three ways to use events (2025 rolling CV, `scripts/evaluate_events.py`):

| Approach | What it does | Result |
|---|---|---|
| City-wide event flags | one column per event kind, the same value at all 34 stations | citywide MAE worse in all 4 folds (18.73 → 18.92) |
| `event_nearby` as a GBM feature | 1 at the station next to the venue on event shifts, else 0 | helps a little on event shifts (7.13 → 6.75) |
| **Post-model adjustment** | add the average extra calls to the GBM forecast | **helps most on event shifts (7.13 → 6.17)** |

- **City-wide flags fail** because an event changes calls only next to the venue, not across the city. The flag adds noise at the other 33 stations.
- **The feature is weak** because it is 1 in only about 20 of ~12,000 rows a year. The trees rarely split on it.
- **The adjustment works** because it uses the effect measured directly at that station. The model does not have to discover it.

This is not a per-station model. There is still one GBM for all stations; per-station models were tested earlier and were worse (`scripts/evaluate_by_station.py`).

---

## 2. Which events and stations

### How the pairs were chosen

For each venue, we looked at its 2 nearest stations. For each event kind and station, we compared the calls on event shifts with the same weekday ±4 weeks without that kind of event.

The rule was set before looking at 2025. A pair is kept only if, **on 2022-2024 data only**:

- there were at least 10 events,
- calls were at least 10% higher, and
- the effect is clear (t ≥ 2).

### Result

| Event | Venue | Station | Events (2022-2024) | Usual calls | Extra calls | Extra % | t |
|---|---|---|---|---|---|---|---|
| Husky football | Husky Stadium | SFD 22 | 19 | 4.0 | +4.4 | +111% | 5.95 |
| Seahawks | Lumen Field | SFD 10 | 27 | 36.4 | +3.8 | +11% | 2.19 |
| Stadium concert | Lumen Field | SFD 10 | 17 | 39.1 | +4.1 | +11% | 2.11 |

### Not chosen

These events add only 0-4% at the nearest stations:

- Mariners (T-Mobile Park)
- Sounders (Lumen Field)
- Kraken and Storm (Climate Pledge Arena)
- arena concerts (Climate Pledge Arena)
- stadium concerts at T-Mobile Park (10 events, no clear effect)

Two earlier checks also found no effect:

- Festivals, parades and markets: large permitted events (10,000+ people) show no citywide forecast error in 2023-2024.
- Pride and Seafair, tested separately, show no consistent pattern.

### What this tells us

What matters is a very large crowd in one place, not whether the event is sport or music:

- Lumen Field holds ~68,000 people. Football games and stadium concerts there have the same effect at SFD 10.
- Climate Pledge Arena holds ~17,000 people. Neither games nor concerts there have an effect.

This partly matches Kienbacher et al. (2024), who found an effect for music events but not for sports.

### Where the extra calls happen

A station covers a large area, so a higher count at SFD 10 alone does not prove the calls come from the stadium. Each call in the raw 911 export has a location, so we counted calls by distance from the venue, from 3 hours before to 6 hours after the start. We compared them with the same time on the same weekday ±4 weeks without that kind of event, using 2022-2024 only. Concerts have no start time in setlist.fm, so 19:00 is assumed.

Extra calls per event, by distance from the venue:

| Event | 0-0.5 km | 0.5-1 km | 1-2 km | 2-4 km |
|---|---|---|---|---|
| Husky football | **+2.5** (0.3 → 2.9, t = 5.4) | +0.7 | +1.3 | +1.1 (not clear) |
| Seahawks | **+3.1** (2.6 → 5.7, t = 5.7) | +0.3 | +1.3 (not clear) | +1.5 (not clear) |
| Stadium concert, Lumen Field | **+2.8** (3.3 → 6.1, t = 3.5) | −0.6 | +0.7 | −0.7 |
| Sounders | +1.0 (t = 3.1) | 0.0 | +0.3 | +1.5 |
| Mariners | +1.1 (t = 10.6) | +0.4 | −0.3 | **−2.1** (t = −3.8) |

- **The extra calls are at the venue.** Almost all of the increase is within 0.5 km, the stadium and its parking. Of the +3.8 calls per Seahawks game at SFD 10, +3.1 are within 0.5 km during the game window.
- **Husky football spills over a little**, with +1.3 calls at 1-2 km (the U District and campus).
- **Mariners games move calls rather than add them.** There are more calls at the ballpark, but fewer 2-4 km away. The total in the station barely changes (+3%), which is why the Mariners get no adjustment.
- Lumen Field and T-Mobile Park are only ~0.4 km apart, so their rings overlap.

---

## 3. How the extra calls are computed

For one pair (event kind, station), using **training shifts only**:

1. For each training event shift `d`, take the calls at the station on `d`.
2. Take the same weekday 1-4 weeks before and after `d` (up to 8 shifts). Drop shifts that have the same kind of event, and outage (masked) shifts. Keep the event only if at least 3 remain.
3. The difference for this event = calls on `d` − mean calls on those shifts.
4. **Extra calls** = the mean difference over all training events.

Each comparison shift lies within 4 weeks of its event. So training events from the last 28 days before the training cutoff are left out: their "after" weeks would fall outside the training data.

The extra calls are **recomputed every time the model is retrained**. The numbers in section 2 are from 2022-2024 and will change slightly when 2025 is added. They are not fixed constants.

---

## 4. Results (2025 rolling CV, 4 folds)

| Version | Station MAE | Citywide MAE | MAE on the 19 event-station shifts | Bias on them |
|---|---|---|---|---|
| Current GBM | 2.409 | 18.95 | 7.27 | −4.94 |
| + special days | 2.406 | 18.73 | 7.13 | −4.82 |
| + special days + adjustment | **2.405** | 18.75 | **6.17** | **−0.83** |

Bias is the forecast minus the actual (negative = under-forecast).

- On event shifts at the chosen stations, the error drops by 13%. The under-forecast of almost 5 calls nearly disappears.
- Overall MAE does not change, because event-station shifts are a very small share of all rows (19 of ~12,000 in 2025).
- The gain is local, but it lands where it matters: the busiest station (SFD 10) on its busiest days.

---

## 5. How to use it in the forecast

### Data needed

| File | Made by | Contents |
|---|---|---|
| `data/events.csv` | `scripts/fetch_events.py` | home games: team, venue, start time, ShiftDate |
| `data/concerts.csv` | `scripts/fetch_concerts.py` | concert nights at large venues: venue, size (stadium / arena), ShiftDate |

Both schedules are published weeks to months ahead, so they are known 3 shifts before the forecast (no leakage).

### Steps

At training time (`scripts/train_model.py`):

1. Train the GBM as usual.
2. For each pair in section 2, compute the extra calls on the training data (section 3).
3. Save the table, e.g. in `model_metadata.json`:
   ```json
   "event_adjustment": [
     {"kind": "Husky football",  "venue": "Husky Stadium", "station": "SFD 22", "extra_calls": 4.4},
     {"kind": "Seahawks",        "venue": "Lumen Field",   "station": "SFD 10", "extra_calls": 3.8},
     {"kind": "stadium concert", "venue": "Lumen Field",   "station": "SFD 10", "extra_calls": 4.1}
   ]
   ```

At forecast time (`predict.py`):

1. Predict with the GBM.
2. For each forecast shift and station, look up `events.csv` and `concerts.csv`. If a listed event falls on that shift at that venue, add its `extra_calls` to that station's forecast.
3. The two SFD 10 pairs (Seahawks, stadium concert) share Lumen Field, so they never fall on the same night. `scripts/fetch_concerts.py` already drops "concerts" on game nights (anthem and halftime shows).

The app can show a note on those shifts, e.g. "Seahawks home game: +4 calls expected at SFD 10".

---

## 6. Limitations

- **Few events.** Each pair has only 17-27 training events, so the estimate is uncertain. The Seahawks and concert effects (t ≈ 2.1-2.2) are just above the bar. With more years the estimate becomes more reliable.
- **Unseen events.** The adjustment only covers event kinds seen in training. The **2026 FIFA World Cup** matches at Lumen Field (June-July 2026, in the test period) have no training examples. The forecast for those shifts may be too low. We report this as a known limitation instead of adding a rule after seeing the test results.
- **Concert data.** setlist.fm is user-contributed and lists only shows that took place (cancelled shows are removed). For large venues cancellations are rare. A few small stadium events (e.g. a single local artist) remain in the stadium list. For live forecasting of future shows, an official ticketing source (e.g. Ticketmaster) would be more complete.
- **One number per pair.** The adjustment is the same for every game of a kind. It does not depend on attendance, kickoff time or the opponent. Football games are near capacity every time, and concert attendance is not published, so the venue stands in for crowd size.

---

## 7. Reproduce

From the repository root:

```bash
uv run python scripts/fetch_events.py        # data/events.csv
uv run python scripts/fetch_concerts.py      # data/concerts.csv (needs SETLISTFM_API_KEY in .env, see .env.example)
uv run python scripts/evaluate_events.py     # part 1, part 2 and the CV table above
```

Related experiments with no gain (results at the end of each script):

- `scripts/evaluate_lag_features.py`: citywide, neighbor, trend, volatility, call-type and last-year lags.
- `scripts/evaluate_calendar_features.py`: month start, UW and Seattle Public Schools calendars, long weekends, day length. Only the special days helped.
