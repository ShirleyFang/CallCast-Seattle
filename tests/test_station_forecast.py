"""Prediction tab pieces: the watch list, the map shades, the call type mix, picking stations
on the map or in the table, the ranked table and download, and the freshness line.

Run: uv run python -m unittest discover -s tests -v
"""

import re
import unittest

import pandas as pd

from app_data import (
    GAP_MESSAGE,
    ROOT,
    hourly_allocation,
    known_history,
    learned_predictions,
    load_tables,
    reference_predictions,
    station_hour_peaks,
)
from prediction_cards import (
    TABLE_CSS,
    briefing_html,
    history_legend_html,
    map_legend_html,
    range_bar_html,
    ranked_table_html,
    week_bars_html,
)
from chart_theme import (
    DAY_STOPS, INK, PREDICTION_TONES, ROSE_LINE, history_window_chart, hourly_altair_chart,
    scale_top,
)
from station_forecast import (
    BRIEF_BUSIER_CALLS,
    BRIEF_USUAL_CALLS,
    DIVERGE_STOPS,
    RECENT_DAYS,
    STATION_USUAL_CALLS,
    WATCH_MAX,
    biggest_move,
    data_through,
    difference_fills,
    download_rows,
    history_summary,
    last_weeks,
    mix_table,
    range_bar_spec,
    ranked_rows,
    recent_days,
    same_weekday_calls,
    season_mix,
    shift_briefing,
    station_status_words,
    stations_to_watch,
    table_action,
    toggle_picks,
)


def forecast(rows):
    # (station, difference, usual) -> the columns the watch list and the shading read.
    out = pd.DataFrame(rows, columns=["Station", "difference", "usual_calls"])
    return out.assign(predicted_calls=out["usual_calls"] + out["difference"])


class WatchListTests(unittest.TestCase):
    def test_needs_both_the_calls_and_the_share(self):
        view = forecast(
            [
                ("above", 6, 30),  # 6 calls and 20%: in
                ("small", 1.9, 5),  # 38% but under 2 calls: out
                ("share", 3, 40),  # 3 calls but 7.5%: out
                ("below", -5, 30),
            ]
        )
        self.assertEqual(list(stations_to_watch(view)["Station"]), ["above"])

    def test_exact_threshold_counts(self):
        # Exactly 2 calls and exactly 10% of the usual is in; a hair under either is out.
        view = forecast([("edge", 2.0, 20), ("calls", 1.99, 10), ("share", 2.0, 20.1)])
        self.assertEqual(list(stations_to_watch(view)["Station"]), ["edge"])

    def test_none_above_usual_gives_an_empty_list(self):
        view = forecast([("a", 0.5, 10), ("b", -3, 20)])
        self.assertTrue(stations_to_watch(view).empty)

    def test_shows_the_biggest_gaps_first_and_at_most_five(self):
        view = forecast([(f"s{i}", 3 + i, 10) for i in range(8)])
        shown = list(stations_to_watch(view)["Station"])
        self.assertEqual(len(shown), WATCH_MAX)
        self.assertEqual(shown, ["s7", "s6", "s5", "s4", "s3"])


class ShadeTests(unittest.TestCase):
    def test_blue_below_cream_at_usual_salmon_above(self):
        fills = difference_fills(forecast([("q", -4, 20), ("u", 0, 20), ("b", 4, 20)]))
        self.assertEqual(fills["q"], [98, 160, 176, 210])
        self.assertEqual(fills["u"], [253, 232, 211, 210])
        self.assertEqual(fills["b"], [138, 56, 37, 210])
        # Busier end stays clearly darker than the quieter end so grayscale stays readable.
        luma = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        self.assertGreater(luma(fills["q"]) - luma(fills["b"]), 60)
        self.assertEqual(len(DIVERGE_STOPS), 5)

    def test_both_sides_use_the_same_span(self):
        fills = difference_fills(forecast([("q", -2, 20), ("b", 4, 20)]))
        # +4 sets the span, so -2 is half way down the blue side (the second stop), not the end.
        self.assertEqual(fills["b"], [138, 56, 37, 210])
        self.assertEqual(fills["q"][:3], [153, 205, 216])

    def test_tiny_gaps_do_not_fill_the_whole_scale(self):
        # With every gap under 2 calls, +1 reaches only half way up the salmon side.
        fills = difference_fills(forecast([("a", 1, 20), ("b", -1, 20)]))
        self.assertEqual(fills["a"][:3], [228, 138, 111])
        self.assertNotEqual(fills["a"][:3], [138, 56, 37])


class CallTypeMixTests(unittest.TestCase):
    def setUp(self):
        # Station A has two Octobers in earlier years; B only one. Nov 2025 and Oct 2026
        # are not the target's season or not an earlier year.
        rows = [
            ("A", "2023-10-01", 4, 1, 0, 1),
            ("B", "2023-10-01", 2, 0, 0, 0),
            ("A", "2023-10-02", 6, 1, 0, 1),
            ("B", "2023-10-02", 2, 2, 0, 0),
            ("A", "2025-10-05", 8, 4, 1, 1),
            ("B", "2025-10-05", 2, 0, 1, 0),
            ("A", "2025-11-05", 90, 90, 90, 90),
            ("B", "2025-11-05", 90, 90, 90, 90),
            ("A", "2026-10-05", 90, 90, 90, 90),
            ("B", "2026-10-05", 90, 90, 90, 90),
        ]
        cols = ["Station", "ShiftDate", "medical", "fire", "non_emergency", "other"]
        self.history = pd.DataFrame(rows, columns=cols).assign(
            ShiftDate=lambda d: pd.to_datetime(d["ShiftDate"])
        )

    def test_station_averages_are_for_the_same_months_of_earlier_years(self):
        mix = season_mix(self.history, "2026-10-24")
        # A medical: (4 + 6 + 8) / 3 = 6; fire: (1 + 1 + 4) / 3 = 2.
        self.assertEqual(mix.loc["A", "Medical"], 6)
        self.assertEqual(mix.loc["A", "Fire"], 2)
        self.assertEqual(mix.loc["B", "Medical"], 2)

    def test_citywide_row_adds_the_stations_per_shift(self):
        mix = season_mix(self.history, "2026-10-24")
        # Medical per shift: 6, 8 and 10 across the three shifts: average 8.
        self.assertEqual(mix.loc["Citywide", "Medical"], 8)
        # Fire: 1, 3 and 4 -> 8/3.
        self.assertAlmostEqual(mix.loc["Citywide", "Fire"], 8 / 3)

    def test_table_has_a_row_per_type_and_a_column_per_name(self):
        table = mix_table(season_mix(self.history, "2026-10-24"), ["B", "A"])
        self.assertEqual(list(table.columns), ["B", "A"])
        self.assertEqual(
            list(table.index), ["Medical", "Fire", "Non-emergency", "Other"]
        )
        self.assertEqual(table.loc["Fire", "A"], 2.0)


class PickTests(unittest.TestCase):
    def test_a_click_picks_and_a_second_click_unpicks(self):
        self.assertEqual(toggle_picks([], ["SFD 5"]), (["SFD 5"], []))
        self.assertEqual(toggle_picks(["SFD 5", "SFD 2"], ["SFD 5"]), (["SFD 2"], []))

    def test_picking_keeps_click_order(self):
        picked, _ = toggle_picks(["SFD 10"], ["SFD 2"])
        self.assertEqual(picked, ["SFD 10", "SFD 2"])

    def test_a_fourth_station_is_refused(self):
        picked, refused = toggle_picks(["a", "b", "c"], ["d"])
        self.assertEqual((picked, refused), (["a", "b", "c"], ["d"]))

    def test_a_pick_makes_room_for_the_next_click_in_one_batch(self):
        picked, refused = toggle_picks(["a", "b", "c"], ["a", "d"])
        self.assertEqual((picked, refused), (["b", "c", "d"], []))


class TableTests(unittest.TestCase):
    def view(self):
        # Two stations: SFD 2 is busier, so it is the first row.
        return pd.DataFrame(
            {
                "Station": ["SFD 10", "SFD 2"],
                "predicted_calls": [30.04, 40.017945],
                "usual_calls": [31.2, 38.947368],
                "difference": [-1.16, 1.070577],
                "low": [23, 31],
                "high": [38, 50],
            }
        )

    def test_rows_are_busiest_first_with_the_range_as_words(self):
        weeks = pd.DataFrame({"a": [5, 3], "b": [None, 4]}, index=["SFD 10", "SFD 2"], dtype=object)
        rows = ranked_rows(self.view(), weeks)
        self.assertEqual(list(rows["Station"]), ["SFD 2", "SFD 10"])
        self.assertEqual(list(rows.index), [0, 1])
        self.assertEqual(list(rows["Likely range"]), ["31 to 50", "23 to 38"])
        self.assertEqual(list(rows["Last 4 weeks"]), [[3, 4], [5, None]])
        # One bar column; no numbers column and no usual column.
        self.assertEqual(
            list(rows.columns),
            ["Station", "Expected calls", "Likely range", "Difference", "Last 4 weeks"],
        )

    def test_the_28_days_end_with_the_last_shift_known_at_planning(self):
        # Sat Oct 24 is planned Thu Oct 22 at 08:00, when the Oct 21 shift has just ended.
        days = recent_days("2026-10-24")
        self.assertEqual(days, list(pd.date_range("2026-09-24", "2026-10-21")))
        self.assertEqual(len(days), RECENT_DAYS)
        self.assertEqual(recent_days("2026-11-27")[-1], pd.Timestamp("2026-11-24"))
        self.assertEqual(days[-1], known_history(load_tables()[1], "2026-10-24").ShiftDate.max())

    def test_days_use_real_calls_then_estimates_after_the_data(self):
        days = pd.date_range("2026-10-01", "2026-10-28")
        counts = pd.DataFrame(
            {"Station": "SFD 1", "ShiftDate": days, "calls": range(len(days)), "Masked": False}
        )
        counts.loc[counts.ShiftDate == "2026-10-16", "Masked"] = True
        # Sun Nov 1 shows Oct 2 to Oct 29: Oct 2 is day 1, Oct 16 is an outage and stays empty,
        # Oct 28 is the last day of data and Oct 29 after it takes the estimate.
        dates = recent_days("2026-11-01")
        est = {pd.Timestamp("2026-10-29"): pd.Series({"SFD 1": 12.34, "SFD 2": 5.0})}
        weeks = last_weeks(known_history(counts, "2026-11-01"), dates, days[-1], est, ["SFD 1", "SFD 2"])
        self.assertEqual(list(weeks.loc["SFD 1"]), [*range(1, 15), None, *range(16, 28), 12.3])
        # A station with no counts still gets its estimate and empty real days.
        self.assertEqual(list(weeks.loc["SFD 2"]), [None] * 27 + [5.0])

    def test_days_follow_the_chosen_date(self):
        counts = load_tables()[1]
        ids = sorted(counts.Station.unique())
        oct24 = last_weeks(known_history(counts, "2026-10-24"), recent_days("2026-10-24"),
                           counts.ShiftDate.max(), {}, ids)
        oct23 = last_weeks(known_history(counts, "2026-10-23"), recent_days("2026-10-23"),
                           counts.ShiftDate.max(), {}, ids)
        # SFD 10 on the Saturdays Sep 26 to Oct 17 sit at days 3, 10, 17 and 24.
        self.assertEqual(list(oct24.loc["SFD 10"])[2::7], [36, 46, 37, 41])
        # A day earlier the window moves back one day.
        self.assertEqual(list(oct23.loc["SFD 10"])[1:], list(oct24.loc["SFD 10"])[:-1])

    def test_bars_share_one_scale_and_mark_estimates_gaps_and_the_weekday(self):
        dates = recent_days("2026-11-01")[-4:]
        # Oct 26 to Oct 29 for Sunday Nov 1: no Sunday, so no darker bar.
        html = week_bars_html([10, None, 20, 15.0], dates, [False, False, False, True], 40, 6)
        self.assertIn('class="bar" style="height:25%;--b:#E0E8DE;--e:#9AA39A" title="Mon Oct 26: 10 calls"', html)
        self.assertIn('class="bar gap" title="Tue Oct 27: no data"', html)
        self.assertIn('class="bar est" style="height:38%;--b:#E8E8DA;--e:#989C91" title="Thu Oct 29: about 15.0, an estimate"', html)
        # Colour follows calls on the shared scale: 0 of 40 is mint, 40 of 40 is rose.
        self.assertIn(f"--b:{DAY_STOPS[0]}", week_bars_html([0], dates[:1], [False], 40, 6))
        self.assertIn(f"--b:{DAY_STOPS[-1]}", week_bars_html([40], dates[:1], [False], 40, 6))
        # On a Tuesday shift the Tuesday bar says so in its tooltip and keeps one plain shade;
        # on a Thursday shift the Thursday estimate does too and is drawn lighter too.
        self.assertIn('class="bar" style="height:75%;--b:#EBB2A0;--e:#846E61" '
                      'title="Tue Oct 27, same weekday: 30 calls"', week_bars_html(
                          [10, 30, 20, 15.0], dates, [False] * 3 + [True], 40, 1))
        self.assertIn('class="bar est" style="height:38%;--b:#E8E8DA;--e:#989C91" '
                      'title="Thu Oct 29, same weekday: about 15.0, an estimate"', week_bars_html(
                          [10, None, 20, 15.0], dates, [False] * 3 + [True], 40, 3))
        self.assertNotIn("bar day", html)
        self.assertEqual(len(re.findall(r'class="bar[ "]', html)), 4)

    def test_table_html_says_which_days_are_estimates(self):
        dates = recent_days("2026-11-06")
        weeks = pd.DataFrame({d: [40, 30] for d in dates}, index=["SFD 10", "SFD 2"])
        rows = ranked_rows(self.view(), weeks)
        html = ranked_table_html(rows, ["SFD 10"], dates, pd.Timestamp("2026-10-28"), "2026-11-06")
        self.assertIn("Oct 07 to Nov 03, estimated after Oct 28", html)
        self.assertIn("later days are estimates", html)
        self.assertIn('data-station="SFD 10" class=on', html)
        self.assertIn('data-act="history"', html)
        self.assertIn('data-act="hourly"', html)
        # The Expected bars share one mint-to-rose scale sized to the whole 90 px track.
        self.assertIn(f"--day-scale:linear-gradient(90deg,{','.join(DAY_STOPS)})", html)
        self.assertIn("background: var(--day-scale);\n  background-size: 90px 100%", TABLE_CSS)
        # 28 bars per row; the 4 Fridays in them say "same weekday".
        self.assertEqual(len(re.findall(r'class="bar[ "]', html)), 2 * 28)
        self.assertEqual(html.count(", same weekday: "), 2 * 4)
        self.assertIn("later days are estimates, drawn a little lighter, not real calls.", html)
        # Every day known: the header names the days and says nothing about estimates.
        html = ranked_table_html(rows, [], recent_days("2026-10-24"), pd.Timestamp("2026-10-28"), "2026-10-24")
        self.assertIn("Sep 24 to Oct 21</small>", html)
        self.assertIn("one bar per day from Sep 24 to Oct 21, oldest on the left. Point at a bar "
                      "for its date and calls; Saturdays, the same weekday as this shift, say so.", html)
        self.assertIn("Lighter = fewer calls, rose = more.", html)
        self.assertNotIn("estimate", html)
        self.assertNotIn("<script>", ranked_table_html(
            rows.assign(Station=["<script>", "SFD 10"]), [], recent_days("2026-10-24"), "2026-10-28", "2026-10-24"))

    def test_bars_fill_their_column_in_one_plain_shade(self):
        # Last 4 weeks is 750 px, room for 28 bars of 20 px with 7 px gaps; the bars stretch to
        # fill it, 42 px tall. No stripes and no dark tops: an estimate is only drawn lighter.
        css = " ".join(TABLE_CSS.split())
        self.assertIn("th:last-child, td:last-child { width: 750px; }", css)
        self.assertIn("column-gap: clamp(2px, 1%, 7px); height: 42px; width: 100%; min-width: 194px; }", css)
        self.assertIn("flex: 1 1 0; min-width: 5px; background: var(--b);", css)
        self.assertIn(".bar.est { opacity: 0.7; }", css)
        for old in ["repeating-linear-gradient", ".bar.day", "max-width: 8px", "width: 1%"]:
            self.assertNotIn(old, css)

    def test_last_4_same_weekdays_are_the_last_4_finished_saturdays(self):
        # Oct 24 is planned Thursday Oct 22, so Saturday Oct 17 is the latest Saturday known.
        counts = load_tables()[1]
        recent = same_weekday_calls(known_history(counts, "2026-10-24"), "2026-10-24")
        rows = counts[
            (counts.Station == "SFD 10")
            & counts.ShiftDate.isin(
                pd.to_datetime(["2026-09-26", "2026-10-03", "2026-10-10", "2026-10-17"])
            )
        ]
        self.assertFalse(rows.Masked.any())
        self.assertEqual(recent["SFD 10"], rows["calls"].tolist())
        self.assertEqual(recent["SFD 10"], [36, 46, 37, 41])
        self.assertEqual(len(recent), 34)

    def test_outage_and_later_shifts_stay_out_of_the_same_weekdays(self):
        days = pd.date_range("2026-09-01", "2026-10-31")
        counts = pd.DataFrame(
            {
                "Station": "SFD 1",
                "ShiftDate": days,
                "calls": range(len(days)),
                "Masked": False,
            }
        )
        counts.loc[counts.ShiftDate == "2026-10-17", "Masked"] = True
        recent = same_weekday_calls(known_history(counts, "2026-10-31"), "2026-10-31")
        # Saturdays Sep 19 to Oct 24 are days 18, 25, 32, 39, 46 and 53; Oct 17 (46) is an
        # outage, so it is skipped and not counted as 0. Oct 31 itself is not finished.
        self.assertEqual(recent["SFD 1"], [25, 32, 39, 53])

    def test_each_click_opens_its_own_row_in_any_order(self):
        ids = ["SFD 2", "SFD 10", "SFD 5", "SFD 25", "SFD 31"]
        picks = []
        # Mixed order: windows, ticks, a refused 4th tick, the same window twice, an untick.
        clicks = [
            ("history", "SFD 10", [], None, ("history", "SFD 10")),
            ("tick", "SFD 2", ["SFD 2"], None, None),
            ("history", "SFD 5", ["SFD 2"], None, ("history", "SFD 5")),
            ("hourly", "SFD 25", ["SFD 2"], None, ("hourly", "SFD 25")),
            ("tick", "SFD 10", ["SFD 2", "SFD 10"], None, None),
            ("tick", "SFD 5", ["SFD 2", "SFD 10", "SFD 5"], None, None),
            ("tick", "SFD 31", ["SFD 2", "SFD 10", "SFD 5"], ["SFD 31"], None),
            ("history", "SFD 31", ["SFD 2", "SFD 10", "SFD 5"], None, ("history", "SFD 31")),
            ("history", "SFD 31", ["SFD 2", "SFD 10", "SFD 5"], None, ("history", "SFD 31")),
            ("tick", "SFD 10", ["SFD 2", "SFD 5"], None, None),
            ("hourly", "SFD 2", ["SFD 2", "SFD 5"], None, ("hourly", "SFD 2")),
        ]
        for act, name, want, refused, window in clicks:
            picks, got_refused, got_window = table_action({"act": act, "station": name}, picks, ids)
            self.assertEqual((picks, got_refused or None, got_window), (want, refused, window))
        # No click (the run after a window closed), an unknown station or action: nothing.
        for click in [None, {}, {"act": "history", "station": "SFD 99"}, {"act": "x", "station": "SFD 2"}]:
            self.assertEqual(table_action(click, picks, ids), (picks, [], None))

    def test_history_summary_reads_as_two_plain_lines(self):
        row = pd.Series(
            {"predicted_calls": 40.017945, "low": 31, "high": 50, "usual_calls": 38.947368}
        )
        self.assertEqual(
            history_summary("2026-10-24", [36, 46, 37, 41], row),
            [
                "Last 4 Saturdays: 36, 46, 37, 41.",
                "Expected 40.0 on Sat Oct 24 (likely 31 to 50); usual for a Saturday 38.9.",
            ],
        )
        self.assertEqual(history_summary("2026-10-25", [12], row)[0], "Last Sunday: 12.")
        self.assertEqual(
            history_summary("2026-10-25", [], row)[0], "No finished Sundays are known yet."
        )
        # Far dates add how old the newest known shift is; near ones do not.
        self.assertEqual(len(history_summary("2026-10-24", [1], row, "2026-10-21")), 2)
        self.assertEqual(
            history_summary("2026-12-20", [1], row, "2026-10-28")[2],
            "The newest known shift is Wed Oct 28, 53 days earlier.",
        )


    def test_download_keeps_its_eight_columns_and_values(self):
        csv = download_rows(
            self.view(), "2026-10-24", pd.Timestamp("2026-10-22 08:00-07:00")
        )
        self.assertEqual(
            csv.to_csv(index=False).splitlines(),
            [
                "Station,Expected calls,Low,High,Usual for this weekday,Difference,Shift date,Planning time",
                "SFD 10,30.0,23,38,31.2,-1.2,2026-10-24,2026-10-22T08:00:00-07:00",
                "SFD 2,40.0,31,50,38.9,1.1,2026-10-24,2026-10-22T08:00:00-07:00",
            ],
        )


class HistoryWindowChartTests(unittest.TestCase):
    def calls(self):
        # Sep 1 to Oct 21 with Oct 5 missing (an outage shift).
        days = pd.date_range("2026-09-01", "2026-10-21")
        out = pd.Series(range(len(days)), index=days, dtype=float)
        return out.drop(pd.Timestamp("2026-10-05"))

    def test_4_weeks_shows_the_last_28_days_with_gaps_and_weekday_dots(self):
        fig = history_window_chart(self.calls(), "2026-10-24", 30.0, 40.0, 31, 50, days=28)
        line = next(t for t in fig.data if t.name == "Calls per shift")
        self.assertEqual(pd.Timestamp(line.x[0]), pd.Timestamp("2026-09-24"))
        self.assertEqual(pd.Timestamp(line.x[-1]), pd.Timestamp("2026-10-21"))
        # The missing shift is not a 0: the line skips it, a faint dashed bridge joins Oct 4 and
        # Oct 6, one fill runs through it, and pointing at it names the gap.
        dates = list(pd.to_datetime(line.x))
        self.assertTrue(pd.isna(line.y[dates.index(pd.Timestamp("2026-10-05"))]))
        self.assertFalse(line.connectgaps)
        bridge = next(t for t in fig.data if t.line.dash == "dash")
        self.assertEqual([pd.Timestamp(x) for x in bridge.x[:2]],
                         [pd.Timestamp("2026-10-04"), pd.Timestamp("2026-10-06")])
        gap = next(t for t in fig.data if t.name == "No data")
        self.assertEqual([pd.Timestamp(x) for x in gap.x], [pd.Timestamp("2026-10-05")])
        self.assertIn("No data for this shift (data feed gap)", gap.hovertemplate)
        self.assertEqual(len([t for t in fig.data if t.fill == "tozeroy"]), 1)
        # The line is 3 px deep rose and the fill climbs from mint at 0 to rose at the axis top.
        self.assertEqual((line.line.color, line.line.width), (ROSE_LINE, 3))
        grad = next(t for t in fig.data if t.fill == "tozeroy").fillgradient
        self.assertEqual((grad.type, grad.start, grad.stop), ("vertical", 0, fig.layout.yaxis.range[1]))
        self.assertEqual(grad.colorscale[0][1], "rgba(206, 236, 231, 0.3)")
        self.assertEqual(grad.colorscale[-1][1], "rgba(223, 159, 163, 0.8)")
        dots = next(t for t in fig.data if t.name == "Saturdays")
        self.assertEqual((dots.marker.color, dots.textfont.color), (ROSE_LINE, ROSE_LINE))
        self.assertEqual(
            [f"{pd.Timestamp(x):%a %b %d}" for x in dots.x],
            ["Sat Sep 26", "Sat Oct 03", "Sat Oct 10", "Sat Oct 17"],
        )
        self.assertEqual(list(dots.text), ["25", "32", "39", "46"])

    def test_hourly_bars_follow_the_day_scale(self):
        # 08:00 is quiet (mint), 20:00 is the peak (rose, outlined in ink), 14:00 in between.
        hourly = pd.DataFrame({
            "hour": ["08:00", "14:00", "20:00"], "expected_calls": [0.0, 1.0, 2.0],
            "share": [0.0, 1 / 3, 2 / 3],
        })
        spec = hourly_altair_chart(hourly).to_dict()
        rows = spec["layer"][0]
        data = {r["hour"]: r for r in spec["datasets"][spec["data"]["name"]]}
        self.assertEqual(rows["encoding"]["color"], {"field": "fill", "scale": None, "type": "nominal", "legend": None})
        self.assertEqual(data["08:00"]["fill"], DAY_STOPS[0])
        self.assertEqual(data["20:00"]["fill"], DAY_STOPS[-1])
        self.assertEqual((data["20:00"]["edge"], data["20:00"]["edge_width"]), (INK, 1.5))
        self.assertNotEqual(data["14:00"]["edge"], INK)

    def test_the_estimate_sits_on_its_own_panel_with_its_likely_range(self):
        fig = history_window_chart(self.calls(), "2026-10-24", 30.0, 40.0, 31, 50, days=28)
        est = next(t for t in fig.data if t.name == "Expected, likely range")
        self.assertEqual(list(est.x), ["Sat Oct 24"])
        self.assertEqual(est.error_y.array[0], 10)
        self.assertEqual(est.error_y.arrayminus[0], 9)
        self.assertEqual(est.xaxis, "x2")
        usual = next(t for t in fig.data if (t.name or "").startswith("Usual"))
        self.assertEqual(usual.name, "Usual for a Saturday (30.0)")
        self.assertEqual(list(usual.y), [30.0, 30.0])
        # Nothing to drag or zoom: the span buttons do the zooming.
        self.assertFalse(fig.layout.dragmode)
        self.assertTrue(fig.layout.xaxis.fixedrange)
        self.assertGreaterEqual(fig.layout.yaxis.range[1], 50)

    def test_legend_matches_the_view(self):
        near = history_legend_html("2026-10-24", 38.947, tone="busier")
        self.assertIn("Saturdays", near)
        self.assertIn("Usual for a Saturday (38.9)", near)
        self.assertIn(PREDICTION_TONES["busier"]["main"], near)
        far = history_legend_html("2026-10-24", 38.9, long_view=True)
        self.assertIn("4-week average", far)
        self.assertNotIn("Saturdays", far)

    def test_all_fades_the_shifts_and_adds_a_4_week_average(self):
        fig = history_window_chart(self.calls(), "2026-10-24", 30.0, 40.0, 31, 50, days=None)
        names = [t.name for t in fig.data if t.name]
        self.assertIn("4-week average", names)
        self.assertNotIn("Saturdays", names)
        line = next(t for t in fig.data if t.name == "Calls per shift")
        self.assertEqual(pd.Timestamp(line.x[0]), pd.Timestamp("2026-09-01"))


class SharedScaleTests(unittest.TestCase):
    # Oct 24: SFD 16 is a quiet station and SFD 10 a busy one, so their windows must not
    # both peak in rose.
    @classmethod
    def setUpClass(cls):
        stations, counts, features = load_tables()
        ids = stations.station.tolist()
        target = "2026-10-24"
        cls.totals = learned_predictions(features, target, ids)[0].set_index("Station")["predicted_calls"]
        cls.past = known_history(counts, target)
        dates = recent_days(target)
        cls.weeks = last_weeks(cls.past, dates, counts["ShiftDate"].max(), {}, ids)

    def peak_fill(self, name, top):
        hourly = hourly_allocation(ROOT, "2026-10-24", name, self.totals[name])
        spec = hourly_altair_chart(hourly, top).to_dict()
        rows = spec["datasets"][spec["data"]["name"]]
        return max(rows, key=lambda r: r["expected_calls"])["fill"]

    def test_busiest_hour_colours_differ_between_stations(self):
        top = scale_top(station_hour_peaks(ROOT, "2026-10-24", self.totals))
        quiet, busy = self.peak_fill("SFD 16", top), self.peak_fill("SFD 10", top)
        self.assertNotEqual(quiet, busy)
        self.assertNotEqual(quiet, DAY_STOPS[-1])

    def test_history_peaks_share_the_table_scale(self):
        # Both windows' fills reach rose at the table's Last 4 weeks top, not their own peaks.
        top = scale_top(self.weeks.to_numpy().ravel())
        stops = {}
        for name in ("SFD 16", "SFD 10"):
            calls = self.past.loc[self.past["Station"] == name].set_index("ShiftDate")["calls"]
            calls.index = pd.to_datetime(calls.index)
            fig = history_window_chart(calls, "2026-10-24", 10.0, 12.0, 8, 16, scale=top)
            grad = next(t for t in fig.data if t.fill == "tozeroy").fillgradient
            stops[name] = (grad.stop, float(calls.tail(28).max()) / grad.stop)
        self.assertEqual(stops["SFD 16"][0], stops["SFD 10"][0])
        self.assertLess(stops["SFD 16"][1], stops["SFD 10"][1])


class FreshnessTests(unittest.TestCase):
    def test_data_runs_to_the_end_of_the_last_finished_shift(self):
        # The Wednesday shift starts at 08:00 Wednesday and ends at 08:00 Thursday.
        history = pd.DataFrame(
            {"ShiftDate": pd.to_datetime(["2026-10-20", "2026-10-21"])}
        )
        self.assertEqual(data_through(history), "Thu Oct 22 08:00")

    def test_a_shift_with_missing_data_pulls_the_time_back(self):
        history = pd.DataFrame(
            {"ShiftDate": pd.to_datetime(["2026-10-19", "2026-10-20"])}
        )
        self.assertEqual(data_through(history), "Wed Oct 21 08:00")


class BriefingTests(unittest.TestCase):
    def test_briefing_qualitative_thresholds_and_no_percentages(self):
        # Quieter when citywide gap is below -BRIEF_USUAL_CALLS; names stations past quiet rule.
        quiet = forecast([("SFD 10", -(BRIEF_USUAL_CALLS + 1), 38), ("SFD 2", -4, 32)])
        s_quiet = shift_briefing("2026-01-03", quiet)
        self.assertEqual(
            s_quiet, "Saturday looks quieter than usual, led by SFD 10 and SFD 2."
        )
        # A quieter day with a station passing the watch rule still flags that busier station.
        quiet_watch = forecast([("SFD 10", -10, 38), ("SFD 6", 4, 20)])
        self.assertEqual(
            shift_briefing("2026-07-27", quiet_watch),
            "Monday looks quieter than usual, led by SFD 10, though SFD 6 may see more calls.",
        )
        # About usual when within BRIEF_USUAL_CALLS and no watch station.
        usual = forecast([("SFD 10", 1.0, 38), ("SFD 2", -1.0, 32)])
        self.assertEqual(
            shift_briefing("2026-10-27", usual),
            "Tuesday looks about usual across Seattle.",
        )
        # About usual citywide with a single watch station still names it.
        mixed = forecast([("SFD 10", 4.0, 30), ("SFD 2", -3.0, 30)])
        self.assertIn(
            "though SFD 10 may see more calls", shift_briefing("2026-10-24", mixed)
        )
        # A little busier when between BRIEF_USUAL_CALLS and BRIEF_BUSIER_CALLS (names watch stations only).
        mild = forecast([("SFD 10", 4.0, 38), ("SFD 2", 3.5, 32), ("SFD 25", 3.0, 28)])
        self.assertEqual(
            shift_briefing("2026-10-24", mild),
            "Saturday looks a little busier than usual, led by SFD 10, SFD 2 and SFD 25.",
        )
        # Small differences across many stations say "spread across the city".
        spread = forecast([(f"SFD {i}", 1.5, 30) for i in range(1, 6)])
        self.assertEqual(
            shift_briefing("2026-10-24", spread),
            "Saturday looks a little busier than usual, spread across the city.",
        )
        # Exact boundaries stay in the calmer bucket; one notch past moves up.
        self.assertEqual(
            shift_briefing("2026-10-24", forecast([("SFD 1", -BRIEF_USUAL_CALLS, 60)])),
            "Saturday looks about usual across Seattle.",
        )
        self.assertEqual(
            shift_briefing("2026-10-24", forecast([("SFD 1", BRIEF_USUAL_CALLS, 60)])),
            "Saturday looks about usual across Seattle.",
        )
        self.assertEqual(
            shift_briefing(
                "2026-10-24", forecast([("SFD 1", BRIEF_BUSIER_CALLS, 210)])
            ),
            "Saturday looks a little busier than usual, spread across the city.",
        )
        # Busier when above BRIEF_BUSIER_CALLS.
        busy = forecast([("SFD 2", BRIEF_BUSIER_CALLS + 5, 32), ("SFD 10", 5, 38)])
        s_busy = shift_briefing("2027-01-17", busy)
        self.assertTrue(
            s_busy.startswith(
                "Sunday looks busier than usual, led by SFD 2 and SFD 10."
            )
        )
        self.assertNotIn("%", s_quiet + s_busy)

    def test_real_dates_briefing_matches_watch_list(self):
        stations, counts, features = load_tables(ROOT)
        ids = sorted(stations["station"], key=lambda x: int(x.split()[-1]))
        for day, expected in [
            (
                "2026-10-24",
                "Saturday looks a little busier than usual, spread across the city.",
            ),
            (
                "2027-01-17",
                "Sunday looks busier than usual, led by SFD 2, SFD 25 and SFD 5.",
            ),
        ]:
            usual = reference_predictions(counts, day, ids)
            res, _ = learned_predictions(features, day, ids)
            view = res.merge(
                usual.rename(columns={"predicted_calls": "usual_calls"}), on="Station"
            )
            view["difference"] = view["predicted_calls"] - view["usual_calls"]
            self.assertEqual(shift_briefing(day, view), expected)
            if day == "2027-01-17":
                watch_names = set(stations_to_watch(view)["Station"])
                self.assertTrue({"SFD 2", "SFD 25", "SFD 5"}.issubset(watch_names))

    def test_no_estimate_date_gets_a_plain_sentence(self):
        gap = shift_briefing("2026-01-09", None, GAP_MESSAGE)
        self.assertIn("Friday has no call estimate", gap)
        self.assertIn("gap", gap)
        unrelated = shift_briefing("2026-01-09", None, "unexpected gap in config")
        self.assertEqual(unrelated, "Friday has no call estimate available yet.")
        other = shift_briefing("2027-02-01", None, "later")
        self.assertEqual(other, "Monday has no call estimate available yet.")

    def test_station_status_words_and_biggest_move(self):
        self.assertEqual(station_status_words(-2.5), "Quieter than usual")
        self.assertEqual(
            station_status_words(-(STATION_USUAL_CALLS + 0.1)),
            "A little quieter than usual",
        )
        self.assertEqual(station_status_words(0.2), "About usual")
        self.assertEqual(station_status_words(1.1), "A little busier than usual")
        self.assertEqual(station_status_words(3.0), "Busier than usual")
        rise_label, rise_row = biggest_move(
            forecast([("SFD 2", 1, 30), ("SFD 10", 2, 38)])
        )
        self.assertEqual(
            (rise_label, rise_row["Station"]), ("Biggest rise vs usual", "SFD 10")
        )
        drop_label, drop_row = biggest_move(
            forecast([("SFD 2", -7, 30), ("SFD 10", 1, 38)])
        )
        self.assertEqual(
            (drop_label, drop_row["Station"]), ("Biggest drop vs usual", "SFD 2")
        )
        usual_drop_label, usual_drop_row = biggest_move(
            forecast([("SFD 2", -1.5, 30), ("SFD 10", -0.5, 38)])
        )
        self.assertEqual(
            (usual_drop_label, usual_drop_row["Station"]),
            ("Biggest drop vs usual", "SFD 2"),
        )


class RangeBarTests(unittest.TestCase):
    def test_range_bar_positions_stay_in_bounds_and_match_tone(self):
        spec = range_bar_spec(40.0, 31, 50, 38.9)
        self.assertEqual(spec["tone"], "busier")
        self.assertGreater(spec["expected_pct"], spec["usual_pct"])
        self.assertLess(spec["band_left"], spec["expected_pct"])
        self.assertGreater(spec["band_left"] + spec["band_width"], spec["expected_pct"])
        self.assertEqual(range_bar_spec(20.0, 14, 27, 25.0)["tone"], "quieter")
        self.assertEqual(range_bar_spec(20.0, 14, 27, 20.2)["tone"], "usual")
        # Zero lower bound stays non-negative and keeps markers inside 0..100.
        low_spec = range_bar_spec(1.5, 0, 4, 2.5)
        self.assertGreaterEqual(low_spec["band_left"], 0.0)
        self.assertLessEqual(low_spec["expected_pct"], 100.0)

    def test_html_builders_include_words_and_escape_text(self):
        bar = range_bar_html(range_bar_spec(40.0, 31, 50, 38.9))
        self.assertIn("31", bar)
        self.assertIn("50", bar)
        self.assertIn("38.9", bar)
        self.assertIn("Expected", bar)
        self.assertIn("Usual", bar)
        self.assertIn("Likely range", bar)
        self.assertNotIn("<script>", briefing_html("<script>"))
        self.assertIn("Quieter than usual", map_legend_html())
        self.assertIn("Busier than usual", map_legend_html())
        self.assertIn("match their cards", map_legend_html())


if __name__ == "__main__":
    unittest.main()
