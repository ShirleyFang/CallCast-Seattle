# What it does
# Downloads concerts at Seattle's large venues from setlist.fm and saves one row
# per venue and night to data/concerts.csv, in the same layout as data/events.csv.
# Concerts are announced months ahead, so they are known when a forecast is made.
# setlist.fm only lists shows that took place (cancelled shows are removed); for
# big venues this is rare, so it is close to the schedule known in advance.
#
# Needs a free setlist.fm API key in .env (see .env.example):
#   SETLISTFM_API_KEY=...
#
# Run it from the repository root: uv run python scripts/fetch_concerts.py
#

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

FIRST, LAST = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-12-31")   # same years as data/events.csv
# large venues inside Seattle (capacity): the stadiums host a few tours a year, the arena dozens
VENUES = ["Climate Pledge Arena",   # ~17,000
          "Lumen Field",            # ~68,000
          "T-Mobile Park",          # ~47,000
          "Husky Stadium"]          # ~70,000


def api_key():
    """Read SETLISTFM_API_KEY from .env (kept out of git)."""
    with open(".env") as env:
        for line in env:
            if line.startswith("SETLISTFM_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("SETLISTFM_API_KEY is missing in .env")


KEY = api_key()


def get(path, **query):
    """Read one JSON page from the setlist.fm API; waits and retries when rate limited."""
    url = f"https://api.setlist.fm/rest/1.0/{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"x-api-key": KEY, "Accept": "application/json",
                                                   "User-Agent": "callcast-seattle"})
    for attempt in range(5):
        time.sleep(1.5)                                   # stay under the API rate limit
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:                         # no results
                return {}
            if error.code != 429:
                raise
            time.sleep(10 * (attempt + 1))                # rate limited: wait longer, then retry
    raise RuntimeError(f"still rate limited: {url}")


rows = []
for name in VENUES:
    # find the venue's setlist.fm id (exact name, in Seattle)
    found = [v for v in get("search/venues", name=name, cityName="Seattle").get("venue", [])
             if v["name"] == name and v["city"]["name"] == "Seattle"]
    if not found:
        print(f"venue not found: {name}")
        continue
    venue_id = found[0]["id"]

    # setlists come newest first, 20 per page; stop once we pass the first year
    page = 1
    while True:
        data = get(f"venue/{venue_id}/setlists", p=page)
        setlists = data.get("setlist", [])
        for s in setlists:
            rows.append((name, s["artist"]["name"], s["eventDate"]))
        dates = pd.to_datetime([s["eventDate"] for s in setlists], format="%d-%m-%Y")
        if not setlists or dates.min() < FIRST or page * data["itemsPerPage"] >= data["total"]:
            break
        page += 1
    print(f"{name}: {sum(r[0] == name for r in rows)} setlists read")

concerts = pd.DataFrame(rows, columns=["venue", "artist", "date"])
concerts["date"] = pd.to_datetime(concerts["date"], format="%d-%m-%Y")
concerts = concerts[concerts["date"].between(FIRST, LAST)]

# one row per venue and night: the headliner and openers are separate setlists.
# setlist.fm has no start time; concerts are in the evening, so the 08:00-08:00 shift is that date
concerts = (concerts.sort_values(["date", "venue", "artist"])
            .groupby(["venue", "date"], as_index=False)
            .agg(artists=("artist", " / ".join)))
concerts["ShiftDate"] = concerts["date"]

# a "setlist" on the night of a home game at the same venue is an anthem or halftime show
# (e.g. Mike McCready at the Mariners, Lil Jon at the Seahawks); the crowd is the game's,
# which data/events.csv already has, so drop these nights
games = pd.read_csv("data/events.csv", parse_dates=["ShiftDate"])[["venue", "ShiftDate"]].drop_duplicates()
during_game = concerts.merge(games, on=["venue", "ShiftDate"], how="left", indicator=True)["_merge"].eq("both")
print(f"dropped {during_game.sum()} nights with a home game at the same venue")
concerts = concerts[~during_game.to_numpy()]

# stadium concerts (Lumen Field, T-Mobile Park, Husky Stadium) draw 40,000-70,000 people, arena ones ~17,000
concerts["size"] = concerts["venue"].map(lambda v: "arena" if v == "Climate Pledge Arena" else "stadium")
concerts = concerts[["venue", "size", "artists", "ShiftDate"]]
concerts.to_csv("data/concerts.csv", index=False)

print(f"\nsaved data/concerts.csv: {len(concerts)} concert nights")
print(concerts.groupby(["venue", concerts["ShiftDate"].dt.year]).size().unstack(fill_value=0).to_string())
