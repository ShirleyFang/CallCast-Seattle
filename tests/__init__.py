"""Tests for CallCast Seattle. Nothing is downloaded while they run."""

import os

# Read only by `python -m unittest tests.test_app`; `discover -s tests` skips this file, so each
# test file that runs the app sets the same flag too.
os.environ.setdefault("CALLCAST_OFFLINE", "1")
