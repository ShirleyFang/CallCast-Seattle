
import tempfile
import unittest
from pathlib import Path

from event_warnings import get_event_warnings


class TestEventWarnings(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.root = Path(self.temp_dir.name)
        data = self.root / "data"
        data.mkdir()

        (data / "events.csv").write_text(
            "team,venue,start_local,ShiftDate\n"
            "Seahawks,Lumen Field,2025-09-07 13:00:00,2025-09-07\n"
            "Husky football,Husky Stadium,2025-09-06 16:00:00,2025-09-06\n"
        )

        (data / "concerts.csv").write_text(
            "venue,size,artists,ShiftDate\n"
            "Lumen Field,stadium,Test Artist,2025-07-01\n"
        )

    def test_seahawks_warning(self):
        warnings = get_event_warnings(
            "2025-09-07", "SFD 10", self.root
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("Seahawks", warnings[0])

    def test_husky_warning(self):
        warnings = get_event_warnings(
            "2025-09-06", "SFD 22", self.root
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("Husky football", warnings[0])

    def test_concert_warning(self):
        warnings = get_event_warnings(
            "2025-07-01", "SFD 10", self.root
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("Stadium concert", warnings[0])

    def test_unaffected_station(self):
        warnings = get_event_warnings(
            "2025-09-07", "SFD 1", self.root
        )
        self.assertEqual(warnings, [])

    def test_no_event(self):
        warnings = get_event_warnings(
            "2025-01-01", "SFD 10", self.root
        )
        self.assertEqual(warnings, [])

    def test_wrong_station(self):
        warnings = get_event_warnings(
            "2025-09-06", "SFD 10", self.root
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
