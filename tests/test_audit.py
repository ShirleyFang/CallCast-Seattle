"""Shared data tests: the CSVs the app reads, the features built from them,
the numbers we report, and the doc links.

Run: uv run python -m unittest discover -s tests -v
"""

import calendar
import functools
import re
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app_data import ROOT, load_tables


@functools.cache
def tables():
    # Load the real CSVs once and share them between test classes.
    return load_tables()


class DataTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stations, cls.counts, cls.features = tables()
        cls.usable = cls.counts[~cls.counts.Masked]


def us_federal_holidays(years):
    # Fixed-date holidays plus "nth weekday of the month" ones (weekday 0 = Monday).
    def nth(year, month, weekday, n):
        days = [
            date(year, month, d)
            for d in range(1, calendar.monthrange(year, month)[1] + 1)
            if date(year, month, d).weekday() == weekday
        ]
        return days[n - 1] if n > 0 else days[n]

    holidays = set()
    for year in years:
        holidays.update(
            date(year, m, d) for m, d in [(1, 1), (6, 19), (7, 4), (11, 11), (12, 25)]
        )
        holidays.update(
            nth(year, m, w, n)
            for m, w, n in [
                (1, 0, 3),  # MLK Day: 3rd Monday of January
                (2, 0, 3),  # Presidents Day: 3rd Monday of February
                (5, 0, -1),  # Memorial Day: last Monday of May
                (9, 0, 1),  # Labor Day: 1st Monday of September
                (10, 0, 2),  # Columbus Day: 2nd Monday of October
                (11, 3, 4),  # Thanksgiving: 4th Thursday of November
            ]
        )
    return holidays


# Loading: the CSVs have the expected size, and broken copies are rejected.


class CsvLoadingTests(DataTestCase):
    def test_tables_have_expected_size(self):
        # 34 stations x 1,762 shifts, 816 outage rows, 541,319 calls in total.
        self.assertEqual(self.counts.Station.nunique(), 34)
        self.assertEqual(self.counts.ShiftDate.nunique(), 1762)
        self.assertEqual(len(self.counts), 59908)
        self.assertEqual(len(self.usable), 59092)
        self.assertEqual(len(self.features), 59092)
        self.assertEqual(self.counts.calls.sum(), 541319)

    def test_broken_csvs_are_rejected(self):
        # Each case breaks one value in a copy of the real CSVs, in a temp folder.
        files = {
            "sfd_stations.csv": self.stations,
            "station_shift_counts.csv": self.counts,
            "modeling_dataset.csv": self.features,
        }
        cases = [
            ("sfd_stations.csv", "lat", np.inf, "coordinates"),
            ("sfd_stations.csv", "lat", 999, "coordinates"),  # off the planet
            ("station_shift_counts.csv", "calls", -1, "whole numbers"),
            ("station_shift_counts.csv", "ShiftDate", "not a date", ""),
            ("modeling_dataset.csv", "calls", 999, "do not match"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            for name, frame in files.items():
                frame.to_csv(root / "data" / name, index=False)
            load_tables(root)  # the unbroken copy loads fine
            for name, column, value, message in cases:
                broken = files[name].copy()
                broken[column] = broken[column].astype(object)
                broken.loc[0, column] = value
                broken.to_csv(root / "data" / name, index=False)
                with self.assertRaisesRegex(ValueError, message, msg=name):
                    load_tables(root)
                files[name].to_csv(root / "data" / name, index=False)


# Features: recompute modeling_dataset.csv columns from the count table.


class FeatureTests(DataTestCase):
    def test_lag_and_average_features(self):
        # Build each feature again from the count table and compare every row.
        # Outage shifts count as missing, and an average needs a minimum number of values.
        calls = (
            self.counts.assign(value=self.counts.calls.where(~self.counts.Masked))
            .set_index(["Station", "ShiftDate"])
            .value
        )
        f = self.features
        for column, lags, minimum in [
            ("lag3", [3], 1),
            ("lag7", [7], 1),
            ("avg7", range(3, 10), 4),
            ("avg28", range(3, 31), 14),
            ("same_weekday_avg4", [7, 14, 21, 28], 2),
        ]:
            earlier = np.column_stack(
                [
                    calls.reindex(
                        pd.MultiIndex.from_arrays(
                            [f.Station, f.ShiftDate - pd.Timedelta(days=lag)]
                        )
                    ).to_numpy()
                    for lag in lags
                ]
            )
            known = np.isfinite(earlier).sum(axis=1)
            expected = np.divide(
                np.nansum(earlier, axis=1),
                known,
                out=np.full(len(f), np.nan),
                where=known >= minimum,
            )
            np.testing.assert_allclose(
                f[column], expected, equal_nan=True, err_msg=column
            )

    def test_calendar_and_holiday_flags(self):
        f = self.features
        self.assertTrue(f.weekday.eq(f.ShiftDate.dt.dayofweek).all())
        self.assertTrue(f.month.eq(f.ShiftDate.dt.month).all())
        self.assertTrue(f.year.eq(f.ShiftDate.dt.year).all())
        # Flags use the actual holiday date, not the observed weekday off.
        holidays = us_federal_holidays(range(2022, 2028))
        days = f.ShiftDate.dt.date
        self.assertEqual(f.is_holiday.tolist(), [int(d in holidays) for d in days])
        self.assertEqual(
            f.is_holiday_eve.tolist(),
            [int(d + timedelta(days=1) in holidays) for d in days],
        )


# Reported numbers: the values in the README and the proposal.


class ReportedNumberTests(DataTestCase):
    def test_target_distribution(self):
        self.assertEqual(self.usable.calls.median(), 6)
        self.assertEqual(round((self.usable.calls == 0).mean() * 100, 2), 2.13)

    def test_random_example_row(self):
        # seed 42 on the rows without an outage, in file order (not sorted).
        raw = pd.read_csv(
            ROOT / "data/station_shift_counts.csv", parse_dates=["ShiftDate"]
        )
        row = raw.loc[~raw.Masked].sample(1, random_state=42).iloc[0]
        self.assertEqual(
            (row.Station, str(row.ShiftDate.date()), row.calls, row.fire, row.medical),
            ("SFD 18", "2024-11-18", 10, 2, 8),
        )

    def test_training_mean_baseline(self):
        # Always guess the 2022-2024 mean, then score it on 2026.
        d = self.usable
        mean = d.loc[d.ShiftDate < "2025", "calls"].mean()
        error = d.loc[d.ShiftDate >= "2026", "calls"] - mean
        self.assertEqual(round(mean, 2), 9.05)
        self.assertEqual(round(error.abs().mean(), 2), 5.98)  # station MAE
        self.assertEqual(round(np.sqrt((error**2).mean()), 2), 9.11)  # station RMSE
        # Citywide: add up all stations per shift first, then compare.
        city = d.groupby("ShiftDate").calls.sum()
        city_error = city[city.index >= "2026"] - city[city.index < "2025"].mean()
        self.assertEqual(round(city_error.abs().mean(), 1), 23.4)


# Docs: every link and image in docs/*.md points to a real file.


class DocLinkTests(unittest.TestCase):
    def test_links_and_images_exist(self):
        for doc in sorted((ROOT / "docs").glob("*.md")):
            text = doc.read_text()
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if target.startswith(("http:", "https:", "#")):
                    continue
                # Absolute paths only work on my laptop, not on GitHub.
                self.assertFalse(Path(target).is_absolute(), f"{doc.name}: {target}")
                path = doc.parent / re.sub(r"(:\d+|#L\d+)$", "", target)
                self.assertTrue(path.exists(), f"{doc.name}: {target}")
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    with Image.open(path) as image:
                        image.verify()  # fails if the image file is broken


if __name__ == "__main__":
    unittest.main()
