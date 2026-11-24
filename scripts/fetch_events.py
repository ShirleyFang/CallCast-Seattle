# What it does
# Downloads the home games of Seattle's major sports teams from public APIs and
# saves one row per game to data/events.csv, with the shift it falls in.
# Game schedules are published months ahead, so they are known when a forecast
# is made (no leakage), including for 2026.
#
# Run it from the repository root: uv run python scripts/fetch_events.py
#
# Sources (no key needed):
#   Mariners (MLB)            statsapi.mlb.com
#   Kraken (NHL)              api-web.nhle.com
#   Seahawks (NFL), Husky football (NCAA), Sounders (MLS), Storm (WNBA)
#                             site.api.espn.com

import json
import urllib.request

import pandas as pd

# from the first modeling year to the end of the app's forecast horizon
# (last known shift 2026-10-28 + FORECAST_DAYS = 90 in step4_features.py)
FIRST, LAST = pd.Timestamp("2022-01-01"), pd.Timestamp("2027-01-26")
SEASONS = range(2021, 2027)          # a season that starts in 2021 can still have games in 2022;
                                     # the 2026 NFL and 2026-27 NHL seasons run into January 2027
SEATTLE_VENUES = ["Lumen Field", "T-Mobile Park", "Climate Pledge Arena", "Husky Stadium"]


def get(url):
    """Read one JSON page from a public API."""
    request = urllib.request.Request(url, headers={"User-Agent": "callcast-seattle"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


rows = []

# Mariners home games: regular season and postseason (F, D, L, W), without postponed or cancelled games
for year in SEASONS:
    data = get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=136&season={year}&gameType=R,F,D,L,W")
    for day in data["dates"]:
        for game in day["games"]:
            if (game["teams"]["home"]["team"]["id"] == 136
                    and game["status"]["detailedState"] not in ("Postponed", "Cancelled")):
                rows.append(("Mariners", game["gameDate"], game["venue"]["name"]))

# Kraken home games: regular season (2) and playoffs (3); NHL seasons are named like 20232024
for year in SEASONS:
    for game in get(f"https://api-web.nhle.com/v1/club-schedule-season/SEA/{year}{year + 1}")["games"]:
        if game["gameType"] in (2, 3) and game["homeTeam"]["abbrev"] == "SEA":
            rows.append(("Kraken", game["startTimeUTC"], game["venue"]["default"]))

# ESPN teams: regular season (seasontype=2), postseason (3), and the default list (needed for MLS)
ESPN = {"Seahawks": "football/nfl/teams/sea",
        "Husky football": "football/college-football/teams/264",
        "Sounders": "soccer/usa.1/teams/9726",
        "Storm": "basketball/wnba/teams/sea"}
for team, path in ESPN.items():
    for year in SEASONS:
        for season_type in ("&seasontype=2", "&seasontype=3", ""):
            data = get(f"https://site.api.espn.com/apis/site/v2/sports/{path}/schedule?season={year}{season_type}")
            for event in data.get("events", []):
                venue = event["competitions"][0].get("venue", {}).get("fullName", "")
                rows.append((team, event["date"], venue))

events = pd.DataFrame(rows, columns=["team", "start_utc", "venue"]).drop_duplicates()

# keep games played in Seattle only: drops away games and neutral-site bowl games
events = events[events["venue"].isin(SEATTLE_VENUES)]

# UTC -> Seattle time; a game belongs to the 08:00-08:00 shift it starts in
start = pd.to_datetime(events["start_utc"], utc=True, format="ISO8601")
events["start_local"] = start.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
events["ShiftDate"] = (events["start_local"] - pd.Timedelta(hours=8)).dt.normalize()
events = events[events["ShiftDate"].between(FIRST, LAST)]

events = (events[["team", "venue", "start_local", "ShiftDate"]]
          .drop_duplicates(["team", "start_local"])
          .sort_values("start_local")
          .reset_index(drop=True))
events.to_csv("data/events.csv", index=False)

print(f"saved data/events.csv: {len(events)} home games")
print(events.groupby(["team", events["ShiftDate"].dt.year]).size().unstack(fill_value=0).to_string())
print(events.groupby(["team", "venue"]).size().to_string())
