from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def get_event_warnings(target_date, station, root=ROOT):
    """Return event warnings for a selected shift and station."""
    day = pd.Timestamp(target_date).normalize()
    warnings = []

    if station not in ("SFD 10", "SFD 22"):
        return warnings

    games = pd.read_csv(root / "data/events.csv", parse_dates=["ShiftDate"])

    if station == "SFD 10":
        seahawks = (
            games["ShiftDate"].eq(day)
            & games["team"].eq("Seahawks")
            & games["venue"].eq("Lumen Field")
        )

        if seahawks.any():
            warnings.append(
                "Seahawks home game at Lumen Field. "
                "Historical data suggests higher call volume near SFD 10."
            )

        concerts = pd.read_csv(
            root / "data/concerts.csv", parse_dates=["ShiftDate"]
        )

        stadium_concert = (
            concerts["ShiftDate"].eq(day)
            & concerts["venue"].eq("Lumen Field")
            & concerts["size"].eq("stadium")
        )

        if stadium_concert.any():
            warnings.append(
                "Stadium concert at Lumen Field. "
                "Historical data suggests higher call volume near SFD 10."
            )

    elif station == "SFD 22":
        husky = (
            games["ShiftDate"].eq(day)
            & games["team"].eq("Husky football")
            & games["venue"].eq("Husky Stadium")
        )

        if husky.any():
            warnings.append(
                "Husky football home game at Husky Stadium. "
                "Historical data suggests higher call volume near SFD 22."
            )

    return warnings