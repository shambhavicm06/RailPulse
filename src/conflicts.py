"""
Which *other trains* does a delay hit?

The cascade model answers "how late will this train be?". A dispatcher's next
question is "and who else does that hold up?" — that needs the schedule
(:mod:`timetable`) plus a set of operating rules.

Four conflict classes are detected, in decreasing confidence:

======================  ==========================================================
Kind                    Rule
======================  ==========================================================
``FOLLOWING_BLOCK``     B trails A into the same directed section. B may not enter
                        until A has cleared it plus headway, so B is held.
``PLATFORM_REGULATION`` A and B are both at the same station within a platform
                        window; the later arrival is regulated for a berth.
``RAKE_TURNAROUND``     B's journey starts at A's destination and A arrives too
                        late for B to depart on time (the rake cannot turn).
``MEET_REGULATION``     A and B cross head-on inside one section, which requires
                        loop regulation **if** the section is single line.
                        Reported as ``assumed: true`` — the network does not carry
                        a single-/double-line attribute, so this one is a
                        sensitivity case, not a finding.
======================  ==========================================================

The delay each affected train absorbs is expressed as a **distribution**, not a
number: A's own calibrated quantiles are propagated to every downstream station,
then the hold is evaluated for each sample. So the response can say "Keerthi
Express: expect +11 min, P(> 15 min) = 0.34" — which is what a regulation
decision actually needs.

Operating constants (headway, block release, regulation penalty, recovery,
platform window, turn-around buffer) are **assumptions**, collected in
:data:`OPERATING_RULES`, reported by the API, and covered by tests. They set the
magnitude of a conflict; the timetable sets whether it exists at all. Both are
derived from simulated data, so results are a demonstration of the mechanism, not
a claim about actual South Western Railway punctuality.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config import DAY_ENC, DAYS
from graph_utils import build_graph, shortest_path_km
from timetable import Timetable, dwell_for, load_timetable, run_time_min

#: Operating assumptions. Every one is a modelling choice, stated so a reader can
#: disagree with a specific number instead of with the whole feature.
OPERATING_RULES = {
    "headway_min": 3.0,
    "block_release_factor": 0.35,
    "platform_window_min": 5.0,
    "platform_regulation_min": 3.0,
    "meet_regulation_min": 6.0,
    "turnaround_buffer_min": 25.0,
    "recovery_factor": 0.6,
    "severe_added_delay_min": 20.0,
    "moderate_added_delay_min": 8.0,
    "threshold_for_probability_min": 15.0,
}

#: How the delayed train's delay grows over the remaining route. ``f`` is the
#: fraction of remaining km already covered, so 0 = here, 1 = destination:
#: ``delay(f) = now + (destination − now) · f ** CURVE``. A curve below 1 front-loads
#: the growth (delay accumulates early), which is the conservative choice.
DELAY_GROWTH_CURVE = 0.85

#: Standard confidence levels used to rebuild A's predictive distribution from the
#: calibrated bounds it already publishes.
_DISTRIBUTION_ANCHORS = {
    0.025: ("95", "lower"), 0.05: ("90", "lower"), 0.10: ("80", "lower"),
    0.25: ("50", "lower"), 0.50: ("median", None),
    0.75: ("50", "upper"), 0.90: ("80", "upper"), 0.95: ("90", "upper"),
    0.975: ("95", "upper"),
}


# ---------------------------------------------------------------------------
# The delayed train's own downstream trajectory
# ---------------------------------------------------------------------------
def _distribution(uncertainty: dict) -> np.ndarray:
    """Ascending (probability, minutes) anchors from the calibrated intervals.

    This is the inverse CDF the Monte Carlo samples from — piecewise linear
    between published quantiles, which is exactly as much resolution as the nine
    quantile models actually provide. It is an approximation of the predictive
    distribution, not a second model.
    """
    intervals = uncertainty.get("intervals", {})
    probs, values = [], []
    for probability, (key, side) in sorted(_DISTRIBUTION_ANCHORS.items()):
        if key == "median":
            value = uncertainty.get("median")
        else:
            band = intervals.get(key) or {}
            value = band.get(side)
        if value is None:
            continue
        probs.append(probability)
        values.append(float(value))
    if len(probs) < 3:                      # uncertainty unavailable: degenerate
        point = float(uncertainty.get("median", 0.0))
        return np.array([point]), np.array([1.0])
    probabilities = np.asarray(probs, dtype=float)
    return probabilities, np.asarray(values, dtype=float)


def _sample_destination_delay(uncertainty: dict, samples: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Draw destination-arrival delays from A's calibrated distribution."""
    probabilities, values = _distribution(uncertainty)
    if values.size == 1:
        return np.full(samples, float(values[0]))
    draws = rng.random(samples)
    return np.interp(draws, probabilities, values)


def _trajectory(entry, current_station: str, destination: str,
                graph, train_type: str,
                delay_samples: np.ndarray,
                anchor_hour: float = 0.0,
                current_delay_min: float = 0.0) -> dict:
    """Where the delayed train will be, and when, along its remaining route.

    Returns per-station scheduled minutes, cumulative km, and the sampled arrival
    delay at each station. Only stations *at or beyond* ``current_station`` are
    included — the past is not a forecast.
    """
    stops: list[str] = []
    scheduled: list[float] = []
    cumulative: list[float] = []

    if entry is not None:
        index = entry.index_of(current_station)
        if index is None:
            # The named train is not recorded at that station; fall back to route
            # geometry rather than inventing a schedule offset.
            index = 0
        remaining = list(entry.stops[index:])
        stops = [stop.station for stop in remaining]
        scheduled = [stop.minute for stop in remaining]
        base_km = entry.path_km[index]
        cumulative = [km - base_km for km in entry.path_km[index:]]

    if len(stops) < 2:
        path = shortest_path_km(graph, current_station, destination) or [current_station]
        if path[-1] != destination and destination in graph:
            path.append(destination)
        stops = path
        scheduled = [0.0]
        cumulative = [0.0]
        for index in range(len(path) - 1):
            km = float(graph.edges[path[index], path[index + 1]]["km"])
            scheduled.append(scheduled[-1] + dwell_for(train_type)
                             + run_time_min(km, train_type))
            cumulative.append(cumulative[-1] + km)

    # Re-anchor the schedule on the hour the caller is asking about: the timetable
    # supplies the *spacing* between stations (that is what the dataset anchors
    # reliably), while "now" is the hour the dispatcher typed. Without this, a
    # train whose recorded slot at this station was 09:33 would be analysed at
    # 09:33 even when the question is about 18:00.
    scheduled = [minute + (anchor_hour * 60.0 - scheduled[0]) for minute in scheduled]

    total_km = max(cumulative[-1], 1e-6)
    fractions = np.clip(np.asarray(cumulative, dtype=float) / total_km, 0.0, 1.0)
    # "Now" is the delay the dispatcher reported, *not* a draw from the
    # destination-delay distribution — mixing those two started the clock at the
    # destination's delay and pushed the whole analysis ~37 minutes into the future.
    now_delay = float(current_delay_min)
    if not delay_samples.size:
        delay_samples = np.array([now_delay])
    # delay[station, sample]
    growth = fractions[:, None] ** DELAY_GROWTH_CURVE
    delays = now_delay + (delay_samples[None, :] - now_delay) * growth
    return {
        "stops": stops,
        "scheduled_min": np.asarray(scheduled, dtype=float),
        "km_remaining": np.asarray(cumulative, dtype=float),
        "delays": delays,                    # shape (stations, samples)
        "median_delay": np.median(delays, axis=1),
    }


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------
def _section_run_min(graph, u: str, v: str, train_type: str) -> float:
    if not graph.has_edge(u, v):
        return 0.0
    return run_time_min(float(graph.edges[u, v]["km"]), train_type)


def _severity(added: float) -> str:
    if added >= OPERATING_RULES["severe_added_delay_min"]:
        return "severe"
    if added >= OPERATING_RULES["moderate_added_delay_min"]:
        return "moderate"
    return "minor"


def _format_minute(minute: float) -> str:
    minute = float(minute) % 1440.0
    return f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"


def affected_trains(*, predictor, current_station: str, destination: str,
                    train_type: str = "Express", hour: int = 12,
                    day: int | str = 0, weather: str = "Clear",
                    current_delay_min: float = 30.0,
                    train_number: Optional[str] = None,
                    horizon_min: float = 120.0,
                    max_trains: int = 25, samples: int = 400,
                    include_meets: bool = False,
                    seed: int = 7,
                    timetable: Optional[Timetable] = None) -> dict:
    """Trains that a delay at ``current_station`` is expected to affect.

    ``train_number`` is optional: when it matches a recorded train the analysis
    uses that train's own path and schedule; otherwise the route is taken from
    ``current_station`` to ``destination``. Either way the schedule is re-anchored
    on ``hour`` — that is "now", and the timetable supplies the spacing.

    ``include_meets`` adds head-on crossings, which only matter on single-line
    sections. The network carries no single-/double-line attribute, so those rows
    are marked ``assumed`` and are **off by default**: findings should not rest on
    infrastructure data this project does not have.
    """
    graph = build_graph()
    routes = timetable if timetable is not None else load_timetable()
    rng = np.random.default_rng(seed)

    # `day` arrives either as a name ("Monday", what /predict/manual takes) or as
    # the dataset's index (0–6, what the timetable is keyed on). Normalise once:
    # passing the wrong one silently analysed every train on a Monday, and passing
    # the index where the model expects a name encoded every day as Monday too.
    day_name = day if isinstance(day, str) else DAYS[int(day) % 7]
    day_index = DAY_ENC.get(day_name, 0)

    # --- the delayed train's own forecast ---------------------------------
    prediction = predictor.predict(
        current_station=current_station, upcoming_station=None,
        destination=destination, train_type=train_type, hour=hour, day=day_name,
        weather=weather, current_delay_min=current_delay_min,
    )
    uncertainty = prediction.get("uncertainty", {}) or {}
    point = float(prediction.get("predicted_destination_arrival_delay_min", 0.0))
    delay_samples = _sample_destination_delay(uncertainty, samples, rng)

    entry = None
    match_note = None
    if train_number:
        key = str(train_number).strip()
        candidate = routes.by_id.get(key.upper()) or routes.by_id.get(key)
        if candidate is None:
            match_note = (f"{key} is not in the reconstructed timetable; the route "
                          f"was taken from {current_station} to {destination}.")
        elif current_station not in candidate.path:
            candidate = None
            match_note = (f"{key} is not scheduled through {current_station}; the "
                          f"route was taken from the geography instead.")
        elif candidate.index_of(destination) is None or \
                candidate.index_of(destination) < candidate.index_of(current_station):
            # The train does not reach the requested destination — using its own
            # path would forecast a different journey than the model's.
            served = candidate.destination
            candidate = None
            match_note = (f"{key} is scheduled to {served}, not {destination}; the "
                          f"route was taken from the geography to {destination}.")
        entry = candidate


    trajectory = _trajectory(entry, current_station, destination, graph,
                             train_type, delay_samples,
                             anchor_hour=float(hour) % 24,
                             current_delay_min=float(current_delay_min))
    stops = trajectory["stops"]
    scheduled = trajectory["scheduled_min"]
    delays = trajectory["delays"]                # (stations, samples)
    base_delay = float(delays[0, 0])

    # Absolute (sampled) time at each station: scheduled + that sample's delay.
    arrival = scheduled[:, None] + delays        # (stations, samples)
    arrival_now = float(np.median(arrival[0]))
    # "In the next N minutes", measured from now — not from the destination
    # arrival, which on a long run would quietly turn a 2-hour question into a
    # 12-hour one.
    window_end = arrival_now + horizon_min

    # A conflict only counts if it happens from now until the end of the horizon.
    # Without these bounds every train that ever uses a section — on any day, at
    # any hour — qualifies, which inflated the first version of this analysis to
    # 826 "affected" trains including rakes said to be waiting 1,772 minutes.
    window_start = arrival_now - 5.0
    # A follower can still be caught even after A has cleared: the tail of A's own
    # delay distribution is long, so allow that much smear past the section.
    block_smear_min = 30.0

    def _in_window(minute: float) -> bool:
        return window_start <= minute <= window_end

    candidates_seen = 0
    threshold = OPERATING_RULES["threshold_for_probability_min"]
    recovery = OPERATING_RULES["recovery_factor"]
    headway = OPERATING_RULES["headway_min"]
    findings: dict[str, dict] = {}

    def _record(other, kind: str, where: str, when: float, hold_samples: np.ndarray,
                reason: str, assumed: bool = False) -> None:
        """Keep the worst conflict per train — one row per affected train."""
        nonlocal candidates_seen
        candidates_seen += 1
        hold_samples = np.clip(hold_samples, 0.0, None)
        if not np.any(hold_samples > 0.5):
            return
        expected = float(np.mean(hold_samples)) * recovery
        if expected < 1.0:
            return
        probability = float(np.mean(hold_samples * recovery > threshold))
        previous = findings.get(other.train_id)
        if previous and previous["expected_added_delay_min"] >= expected:
            return
        findings[other.train_id] = {
            "train_id": other.train_id,
            "train_type": other.train_type,
            "route": f"{other.origin} → {other.destination}",
            "kind": kind,
            "where": where,
            "when": _format_minute(when),
            "expected_added_delay_min": round(expected, 1),
            "p_over_15min": round(probability, 3),
            "severity": _severity(expected),
            "reason": reason,
            "assumed": assumed,
        }

    # --- 1. following conflicts, section by section ------------------------
    for index in range(len(stops) - 1):
        u, v = stops[index], stops[index + 1]
        if u not in graph or v not in graph or not graph.has_edge(u, v):
            continue
        run_min = _section_run_min(graph, u, v, train_type)
        # A occupies the section from its sampled entry to entry + run; a follower
        # may enter only after A clears plus headway.
        a_entry = arrival[index]                       # per sample
        a_clear = a_entry + run_min + headway
        for other, b_entry, _b_exit in routes.section_traffic(u, v, day=day_index):
            if entry is not None and other.train_id == entry.train_id:
                continue
            if not _in_window(b_entry):
                continue
            # A follower enters *at or after* the delayed train. A train that got
            # into the section first is ahead of A, so A waits for it — it cannot
            # be held *by* A. (The previous tolerance of one headway wrongly
            # counted trains that had just gone ahead.)
            if b_entry < float(np.median(a_entry)):
                continue
            if b_entry > float(np.median(a_entry)) + run_min + block_smear_min:
                continue
            hold = a_clear - b_entry                   # broadcast over samples
            _record(other, "FOLLOWING_BLOCK", f"{u} → {v}", b_entry, hold,
                    f"{other.train_id} trails the delayed train into the "
                    f"{u}–{v} block and cannot enter until it clears (+{headway:.0f} "
                    f"min headway).")

    # --- 2. platform / junction regulation at shared stations --------------
    platform_window = OPERATING_RULES["platform_window_min"]
    penalty = OPERATING_RULES["platform_regulation_min"]
    for index, station in enumerate(stops):
        a_time = arrival[index]
        for other, b_time in routes.at_station(
                station, float(np.quantile(a_time, 0.01)) - platform_window,
                min(float(np.quantile(a_time, 0.99)) + platform_window, window_end),
                day=day_index):
            if entry is not None and other.train_id == entry.train_id:
                continue
            if other.train_id in findings:
                continue                      # already held behind it on a block
            if b_time <= float(np.median(a_time)):
                continue                      # B is already in: A waits, not B
            hold = a_time + penalty - b_time  # sampled
            _record(other, "PLATFORM_REGULATION", station, b_time, hold,
                    f"Both trains need {station} within "
                    f"{platform_window:.0f} min; {other.train_id} is regulated for "
                    f"a berth behind the delayed arrival.")

    # --- 3. rake / turn-around --------------------------------------------
    buffer_min = OPERATING_RULES["turnaround_buffer_min"]
    terminus = stops[-1]
    a_arrival = arrival[-1]
    for other, b_depart in routes.departing_from(terminus, day=day_index):
        if entry is not None and other.train_id == entry.train_id:
            continue
        if other.train_id in findings:
            continue
        if not _in_window(b_depart):
            continue
        # A rake can only be shared with a service that departs *after* this train
        # arrives: a train that left this station earlier cannot be waiting for a
        # rake that has not yet turned up.
        if b_depart < float(np.median(a_arrival)):
            continue
        hold = a_arrival + buffer_min - b_depart
        if float(np.median(hold)) <= 0:
            continue
        _record(other, "RAKE_TURNAROUND", terminus, b_depart, hold,
                f"{other.train_id} starts at {terminus} from the same rake; the "
                f"delayed arrival leaves less than the {buffer_min:.0f}-min "
                f"turn-around buffer before departure.")

    # --- 4. head-on meets (assumed: requires a single-line section) --------
    if include_meets:
        meet_penalty = OPERATING_RULES["meet_regulation_min"]
        for index in range(len(stops) - 1):
            u, v = stops[index], stops[index + 1]
            run_min = _section_run_min(graph, u, v, train_type)
            for other, b_entry, b_exit in routes.section_traffic(v, u, day=day_index):
                if entry is not None and other.train_id == entry.train_id:
                    continue
                if other.train_id in findings:
                    continue
                if not _in_window(b_entry):
                    continue
                # Overlap of the two trains' occupation of the same piece of line.
                a_in, a_out = arrival[index], arrival[index] + run_min
                overlap = np.minimum(a_out, b_exit) - np.maximum(a_in, b_entry)
                if float(np.median(overlap)) <= 0:
                    continue
                hold = np.full_like(a_in, meet_penalty)
                _record(other, "MEET_REGULATION", f"{u} ↔ {v}", b_entry, hold,
                        f"{other.train_id} runs the opposite way through the "
                        f"{u}–{v} section while the delayed train is in it; a loop "
                        f"crossing is required **if that section is single line**.",
                        assumed=True)

    # --- rank and summarise ------------------------------------------------
    ordered = sorted(findings.values(),
                     key=lambda row: (row["expected_added_delay_min"],
                                      row["p_over_15min"]), reverse=True)
    by_severity = {"severe": 0, "moderate": 0, "minor": 0}
    by_kind: dict[str, int] = {}
    for row in ordered:
        by_severity[row["severity"]] += 1
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1

    distribution = _distribution(uncertainty)
    return {
        "delayed_train": {
            "train_number": (entry.train_id if entry else (train_number or None)),
            "matched_scheduled_train": entry is not None,
            "schedule_note": match_note,
            "train_type": train_type,
            "at": current_station,
            "destination": destination,
            "current_delay_min": round(float(current_delay_min), 1),
            "predicted_destination_delay_min": round(point, 1),
            "predicted_delay_range_min": [
                round(float(distribution[1][0]), 1),
                round(float(distribution[1][-1]), 1)],
            "arrives_destination_hour": _format_minute(float(np.median(arrival[-1]))),
            "route_length_km": round(float(trajectory["km_remaining"][-1]), 1),
        },
        "affected_count": len(ordered),
        "by_severity": by_severity,
        "by_kind": by_kind,
        "assumed_conflicts": sum(1 for row in ordered if row["assumed"]),
        "affected_trains": ordered[:max_trains],
        "horizon": {
            "minutes": horizon_min,
            "measured_from": "now (the train's current position and delay)",
            "analysed_from": _format_minute(window_start),
            "analysed_until": _format_minute(window_end),
            "day": day_index,
            "day_name": day_name,
            "trains_in_timetable": len(routes),
            "candidates_considered": candidates_seen,
        },
        "rule_set": OPERATING_RULES,
        "assumptions": {
            **routes.assumptions,
            "on_time_candidates": ("other trains are assumed to be running to their "
                                   "scheduled times; only the named train is delayed"),
            "delay_growth_curve": DELAY_GROWTH_CURVE,
            "recovery_factor": recovery,
            "meets_included": include_meets,
            "probability_note": ("P(> 15 min) comes from propagating the calibrated "
                                 "quantiles of the delayed train through the same "
                                 "rules; it is a scenario spread, not an observed "
                                 "frequency"),
        },
    }
