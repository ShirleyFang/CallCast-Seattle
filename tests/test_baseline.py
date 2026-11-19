"""The shared baseline: feature_sets.BASELINE_NUMERIC is the no-weather model of Steps 9 and 15.

Run: uv run python -m unittest discover -s tests -v
"""

import ast
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cv_utils import DATE_COL, STATION_COL, TARGET, build_features  # noqa: E402
from feature_sets import BASELINE_NUMERIC  # noqa: E402

DATA = ROOT / "data"
STEP_SCRIPTS = ["step9_weather_compare.py", "step15_weather_extremes.py"]
WEATHER = ["precip_sum", "precip_max_hr", "snow_sum", "temp_max", "temp_min", "gust_max"]
WEATHER_PREFIXES = ("era5_", "fc_", "wx_", "flag_")


class BaselineTests(unittest.TestCase):
    def test_columns_exist_in_modeling_dataset(self):
        columns = pd.read_csv(DATA / "modeling_dataset.csv", nrows=0).columns
        self.assertEqual([c for c in BASELINE_NUMERIC if c not in columns], [])

    def test_no_weather_target_or_category_columns(self):
        counts = pd.read_csv(DATA / "station_shift_counts.csv", nrows=0).columns
        categories = [c for c in counts if c not in (STATION_COL, DATE_COL, TARGET, "OutageHours", "Masked")]
        self.assertTrue(categories)   # fire, medical, ...
        self.assertNotIn(TARGET, BASELINE_NUMERIC)
        self.assertEqual([c for c in BASELINE_NUMERIC if c in categories], [])
        self.assertEqual([c for c in BASELINE_NUMERIC if c in WEATHER or c.startswith(WEATHER_PREFIXES)], [])

    def test_steps_9_and_15_import_baseline_numeric(self):
        for name in STEP_SCRIPTS:
            with self.subTest(script=name):
                tree = ast.parse((ROOT / "scripts" / name).read_text())
                imported = [a.name for node in ast.walk(tree)
                            if isinstance(node, ast.ImportFrom) and node.module == "feature_sets"
                            for a in node.names]
                self.assertIn("BASELINE_NUMERIC", imported)
                # no hard-coded copy of the list left in the script
                copies = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.List)
                          and {getattr(e, "value", None) for e in node.elts} & set(BASELINE_NUMERIC)]
                self.assertEqual(copies, [])

    def test_weather_dataset_without_weather_gives_baseline(self):
        # the no-weather features Steps 9 and 15 build: year and the era5_/fc_ columns excluded
        df = pd.read_csv(DATA / "modeling_dataset_weather.csv", nrows=1000)
        weather = [c for c in df.columns if c.startswith(("era5_", "fc_"))]
        self.assertEqual(len(weather), 2 * len(WEATHER))
        df["weekday"] = pd.to_datetime(df[DATE_COL]).dt.dayofweek
        _, numeric = build_features(df, extra_exclude=["year"] + weather)
        self.assertEqual(sorted(numeric), sorted(BASELINE_NUMERIC))


if __name__ == "__main__":
    unittest.main()
