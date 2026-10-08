"""
Affected-trains tests.

Two kinds of claim are checked here:

* **Timetable mechanics** — the reconstructed schedule is self-consistent
  (times increase along a path, km increase, windows wrap past midnight
  correctly). Built from an in-memory payload so this stays fast.
* **Conflict invariants** — the properties the first implementation got wrong:
  nothing outside the stated window, no self-conflict, no absurd magnitudes, and
  assumed-infrastructure findings flagged and off by default.
"""
from __future__ import annotations

import pytest

from conflicts import OPERATING_RULES, affected_trains
from timetable import Timetable

# ---------------------------------------------------------------------------
# A tiny timetable: three stations in a line, trains at known times.
# ---------------------------------------------------------------------------
PAYLOAD = {
    "assumptions": {"source": "test"},
    "count": 3,
    "trains": [
        # fast train, runs A -> B -> C
        {"train_id": "T1", "train_type": "Superfast", "day_of_week": 0,
         "origin": "A", "destination": "C", "anchor_station": "A",
         "anchor_minute": 600.0, "stations": ["A", "B", "C"],
         "minutes": [600.0, 645.0, 700.0], "km": [0.0, 60.0, 140.0]},
        # follower, 20 minutes behind
        {"train_id": "T2", "train_type": "Express", "day_of_week": 0,
         "origin": "A", "destination": "C", "anchor_station": "A",
         "anchor_minute": 620.0, "stations": ["A", "B", "C"],
         "minutes": [620.0, 668.0, 726.0], "km": [0.0, 60.0, 140.0]},
        # a train that starts at C (rite of turn-around)
        {"train_id": "T3", "train_type": "Express", "day_of_week": 0,
         "origin": "C", "destination": "A", "anchor_station": "C",
         "anchor_minute": 720.0, "stations": ["C", "B", "A"],
         "minutes": [720.0, 765.0, 810.0], "km": [0.0, 80.0, 140.0]},
    ],
}


def test_timetable_indexes_and_windows():
    tt = Timetable(PAYLOAD)
    assert len(tt) == 3
    assert tt.by_id["T1"].minute_at("B") == 645.0
    assert tt.by_id["T1"].minute_at("Z") is None

    at_b = tt.at_station("B", 640, 700)
    assert [entry.train_id for entry, _ in at_b] == ["T1", "T2"]

    # A journey that runs past midnight: A at 23:50, B at 00:10, C at 00:30.
    # Times are "minutes after the journey's first midnight", so B and C exceed
    # 1440 and must be findable both as late-evening and as early-morning minutes.
    midnight_payload = {"trains": [dict(PAYLOAD["trains"][0], train_id="T4",
                                        minutes=[1430.0, 1450.0, 1470.0])]}
    wrapped = Timetable(midnight_payload)
    assert wrapped.at_station("A", 1425, 1445)     # 23:50
    assert wrapped.at_station("B", 1440, 1460)     # 00:10, as stored
    assert wrapped.at_station("B", 5, 15)          # 00:10, wrapped back to a clock time
    # A train at 23:50 is *not* in the 00:00–00:30 window: that is the previous
    # day's 23:50, and the window is a real clock interval.
    assert wrapped.at_station("A", 0, 30) == []

    followers = tt.section_traffic("A", "B")
    assert [entry.train_id for entry, _, _ in followers] == ["T1", "T2"]
    assert tt.section_traffic("B", "A")[0][0].train_id == "T3"

    starters = tt.departing_from("C")
    assert [entry.train_id for entry, _ in starters] == ["T3"]


def test_timetable_rejects_degenerate_paths():
    """A one-stop journey cannot be scheduled and must be dropped, not crashed on."""
    from timetable import _schedule_for

    from graph_utils import build_graph

    graph = build_graph()
    assert _schedule_for(graph, {"origin": "Mysuru", "current_station": "Nowhere",
                                 "destination": "Mysuru", "train_type": "Express",
                                 "scheduled_hour": 10}) is None


# ---------------------------------------------------------------------------
# Conflict invariants (these are the properties the first version violated)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def analysis(predictor_module):
    return affected_trains(
        predictor=predictor_module, current_station="Mysuru",
        destination="KSR Bengaluru", train_type="Superfast", hour=18, day="Monday",
        weather="Clear", current_delay_min=45.0, horizon_min=120.0)


@pytest.fixture(scope="module")
def predictor_module():
    from inference import CascadePredictor
    return CascadePredictor()


def test_window_bounds_are_reported_and_hold(analysis):
    """Every reported conflict must fall inside the stated window.

    The first implementation compared trains on any day at any hour and produced
    826 "affected" trains, including a rake said to be waiting 1,772 minutes.
    """
    from datetime import datetime

    horizon = analysis["horizon"]
    start = datetime.strptime(horizon["analysed_from"], "%H:%M")
    end = datetime.strptime(horizon["analysed_until"], "%H:%M")
    assert horizon["minutes"] == 120
    assert "now" in horizon["measured_from"]

    for row in analysis["affected_trains"]:
        when = datetime.strptime(row["when"], "%H:%M")
        if end < start:                      # window crosses midnight
            assert when >= start or when <= end, row
        else:
            assert start <= when <= end, row


def test_no_absurd_or_self_conflicts(analysis):
    delayed = analysis["delayed_train"]
    ids = {row["train_id"] for row in analysis["affected_trains"]}
    assert delayed["train_number"] not in ids
    assert len(ids) == len(analysis["affected_trains"]), "duplicate train rows"
    assert analysis["affected_count"] == len(analysis["affected_trains"])

    for row in analysis["affected_trains"]:
        # A hold cannot exceed the delay the delayed train is actually carrying,
        # plus the headway that gets added on top, by an order of magnitude.
        assert 0 < row["expected_added_delay_min"] <= 300, row
        assert 0.0 <= row["p_over_15min"] <= 1.0, row
        assert row["severity"] in {"severe", "moderate", "minor"}
        # severity must agree with the stated thresholds
        expected_severity = ("severe" if row["expected_added_delay_min"] >= OPERATING_RULES["severe_added_delay_min"]
                             else "moderate" if row["expected_added_delay_min"] >= OPERATING_RULES["moderate_added_delay_min"]
                             else "minor")
        assert row["severity"] == expected_severity, row


def test_ordering_and_counts_agree(analysis):
    counts = {"severe": 0, "moderate": 0, "minor": 0}
    for row in analysis["affected_trains"]:
        counts[row["severity"]] += 1
    # by_severity summarises the same rows that were returned
    assert counts == analysis["by_severity"]
    delays = [row["expected_added_delay_min"] for row in analysis["affected_trains"]]
    assert delays == sorted(delays, reverse=True), "rows must be ranked by impact"
    assert sum(analysis["by_kind"].values()) == analysis["affected_count"]
    assert analysis["assumed_conflicts"] == sum(
        1 for row in analysis["affected_trains"] if row["assumed"])


def test_assumed_infrastructure_is_off_by_default_and_flagged_when_on(analysis, predictor_module):
    """Head-on meets depend on a single-line section; the network does not say
    which sections those are, so they must not lead the answer."""
    assert "MEET_REGULATION" not in analysis["by_kind"]

    with_meets = affected_trains(
        predictor=predictor_module, current_station="Mysuru",
        destination="KSR Bengaluru", train_type="Superfast", hour=18, day="Monday",
        weather="Clear", current_delay_min=45.0, horizon_min=120.0,
        include_meets=True)
    meets = [row for row in with_meets["affected_trains"]
             if row["kind"] == "MEET_REGULATION"]
    for row in meets:
        assert row["assumed"] is True, "a meet must be marked as an assumption"
    assert with_meets["assumptions"]["meets_included"] is True


def test_notes_explain_a_train_that_does_not_serve_the_destination(predictor_module):
    """A matched train heading elsewhere must not silently drive the analysis."""
    result = affected_trains(
        predictor=predictor_module, current_station="Mysuru",
        destination="KSR Bengaluru", train_type="Superfast", hour=18, day="Monday",
        weather="Clear", current_delay_min=45.0, train_number="TRN00000")
    delayed = result["delayed_train"]
    assert delayed["matched_scheduled_train"] is False
    assert "Chikkajajur" in (delayed["schedule_note"] or "")


def test_clock_starts_at_the_reported_delay_not_the_forecast(predictor_module):
    """The analysis window opens at 'now' = hour + current delay.

    The first implementation seeded the clock from a destination-delay sample, so
    an 18:00 / 45-min case was analysed from 19:22 instead of 18:45.
    """
    result = affected_trains(
        predictor=predictor_module, current_station="Mysuru",
        destination="KSR Bengaluru", train_type="Superfast", hour=18, day="Monday",
        weather="Clear", current_delay_min=45.0, horizon_min=60.0)
    assert result["horizon"]["analysed_from"] == "18:40"     # 18:45 − 5 min slack
    assert result["horizon"]["analysed_until"] == "19:45"    # 18:45 + 60 min
