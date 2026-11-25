"""Tests for CallCast Seattle. Nothing is downloaded while they run."""

import os

# set before any test module imports the app, so weather_alerts never downloads a forecast
os.environ.setdefault("CALLCAST_OFFLINE", "1")
