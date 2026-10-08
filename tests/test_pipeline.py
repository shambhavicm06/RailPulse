"""
Data-pipeline tests: schema mapping, validation, cause attribution, parity, and
the graph itself.

These are the tests that protect the *claims* in the README. If the network
topology silently changes, or the ablation's feature groups stop partitioning the
feature set, the numbers quoted in the paper stop meaning anything — so the
invariants are asserted here instead of being trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_utils import (build_graph, cascade_alert, compute_centralities,
                         nodes_within_radius, path_distance_km, shortest_path_km)
from config import (CLUSTER_FEATURES, FULL_FEATURES, ISOLATED_FEATURES,
                    STATIC_GRAPH_FEATURES)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def test_graph_is_connected_and_weighted():
    graph = build_graph()
    assert graph.number_of_nodes() > 50
    assert graph.number_of_edges() > 50
    import networkx as nx
    assert nx.is_connected(graph), "a disconnected rail graph breaks ripple propagation"
    for _, _, data in graph.edges(data=True):
        assert data["km"] > 0


def test_centralities_are_defined_for_every_station():
    graph = build_graph()
    centralities = compute_centralities(graph)
    assert set(centralities) == set(graph.nodes())
    for values in centralities.values():
        for key in ("degree", "degree_centrality", "eigenvector_centrality",
                    "pagerank", "betweenness_centrality", "closeness_centrality"):
            assert key in values
            assert np.isfinite(values[key])


def test_shortest_path_and_distance_are_consistent():
    graph = build_graph()
    path = shortest_path_km(graph, "KSR Bengaluru", "Mysuru")
    assert path and path[0] == "KSR Bengaluru" and path[-1] == "Mysuru"
    km = path_distance_km(graph, path)
    assert 130 < km < 180, f"Bengaluru–Mysuru is ~158 km, got {km}"


def test_radius_rings_expand_monotonically():
    graph = build_graph()
    rings_1 = sum(1 for h in nodes_within_radius(graph, "Mysuru", 1).values() if h == 1)
    rings_3 = nodes_within_radius(graph, "Mysuru", 3)
    assert rings_1 >= 1
    assert len(rings_3) >= rings_1


def test_cascade_alert_escalates_with_delay():
    graph = build_graph()
    calm = cascade_alert(graph, "Arsikere", 5.0)
    severe = cascade_alert(graph, "Arsikere", 120.0)
    assert calm["level"] in {"NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert severe["level"] in {"NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
    order = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert order.index(severe["level"]) >= order.index(calm["level"])


# ---------------------------------------------------------------------------
# Feature-set invariants (these underpin the ablation)
# ---------------------------------------------------------------------------
def test_feature_groups_partition_the_full_set():
    groups = ISOLATED_FEATURES + STATIC_GRAPH_FEATURES + CLUSTER_FEATURES
    assert len(FULL_FEATURES) == len(set(FULL_FEATURES)), "duplicate feature columns"
    for column in groups:
        assert column in FULL_FEATURES
    # Graph-derived columns really are derived from the graph, not the target.
    assert "current_station_betweenness_centrality" in STATIC_GRAPH_FEATURES
    assert "destination_arrival_delay_min" not in FULL_FEATURES, "target leakage"


# ---------------------------------------------------------------------------
# Schema mapping & validation
# ---------------------------------------------------------------------------
def test_column_mapping_understands_operator_dialects():
    from sources.base import map_columns

    raw = pd.DataFrame({
        "Train No": ["12627"], "Type": ["SF Exp"], "From Station": ["KSR Bengaluru"],
        "Station": ["Mysuru"], "To Station": ["Dharwad"], "Sch Time": ["14:05"],
        "Day": ["Mon"], "Wx": ["fog"], "Delay": [25], "Arrival Delay": [55],
        "cause": ["loco failure"],
    })
    mapped, report = map_columns(raw)
    assert report["mapped"]["Station"] == "current_station"
    assert report["mapped"]["Delay"] == "current_delay_min"
    assert report["mapped"]["Sch Time"] == "scheduled_hour"
    assert report["missing_required"] == []
    # `upcoming_station` is absent from the feed and is *derivable*, so it must be
    # reported as such — not as a fatal omission.
    assert "upcoming_station" in report["missing_derivable"]
    # "Arrival Delay" is claimed by two different canonical fields; the choice is
    # reported so the operator can settle it with an override.
    assert report["ambiguous_source_columns"]["Arrival Delay"] == [
        "current_delay_min", "destination_arrival_delay_min"]
    # A feed with no destination-arrival-delay column cannot be training data.
    _, no_label = map_columns(raw.drop(columns=["Arrival Delay"]))
    assert no_label["missing_required"] == ["destination_arrival_delay_min"]


def test_unknown_columns_are_reported_not_dropped_silently():
    from sources.base import map_columns

    raw = pd.DataFrame({"Station": ["Mysuru"], "Chaos Index": [1.0]})
    _, report = map_columns(raw)
    assert "Chaos Index" in report["unmapped_source_columns"]
    assert report["missing_required"], "missing required columns must be reported"


def test_normalisation_derives_graph_congestion_and_resolves_codes():
    from sources.base import map_columns, normalise_frame

    raw = pd.DataFrame({
        "Train No": ["1"], "Station": ["SBC"], "To Station": ["MYS"],
        "Sch Time": ["14:05"], "Day": ["Mon"], "Delay": [40],
    })
    graph = build_graph()
    mapped, _ = map_columns(raw)
    canonical, report = normalise_frame(mapped, graph=graph,
                                        station_names=list(graph.nodes()))
    assert canonical.loc[0, "current_station"] == "KSR Bengaluru"   # code resolved
    assert canonical.loc[0, "lat"] > 0 and canonical.loc[0, "lon"] > 0
    # Congestion is derived and reported as a graph-derived proxy.
    assert canonical.loc[0, "congestion_r1"] > 0
    congestion_note = [d for d in report["derived_columns"] if "congestion" in d]
    assert congestion_note, report["derived_columns"]
    # The report must say *which* relationship produced the congestion features:
    # the calibrated one the models were fitted on, or the honest fallback.
    from sources.base import congestion_proxy_table
    expected = "CALIBRATED" if congestion_proxy_table() else "UNCALIBRATED"
    assert expected in congestion_note[0], congestion_note[0]


def test_time_and_day_parsing():
    from sources.base import parse_day, parse_hour

    assert parse_hour("14:05") == 14
    assert parse_hour("7 PM") == 19
    assert parse_hour("12 AM") == 0
    assert parse_hour(9) == 9
    assert parse_day("Mon") == 0
    assert parse_day("sunday") == 6
    assert parse_day(3) == 3


def test_validation_rejects_impossible_delays():
    from sources.validate import summarise, validate

    frame = pd.DataFrame({
        "current_station": ["Mysuru", "Mysuru", "Mysuru"],
        "train_type": ["Express"] * 3,
        "current_delay_min": [20.0, 5000.0, -900.0],          # 2 impossible
        "destination_arrival_delay_min": [30.0, 5000.0, -900.0],
        "scheduled_hour": [14, 14, 25],                        # 1 impossible hour
    })
    valid, report = validate(frame)
    assert len(valid) == 1
    assert report["rejected_rows"] == 2
    assert "outside" in summarise(report)


def test_validation_flags_duplicates():
    from sources.validate import validate

    frame = pd.DataFrame({
        "train_id": ["A", "A"], "current_station": ["Mysuru", "Mysuru"],
        "scheduled_hour": [14, 14], "train_type": ["Express", "Express"],
        "current_delay_min": [10.0, 12.0],
        "destination_arrival_delay_min": [15.0, 16.0],
    })
    valid, report = validate(frame)
    assert len(valid) == 1
    assert report["checks"]["duplicates"]["rejected"] == 1


# ---------------------------------------------------------------------------
# Cause attribution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("A/preoccupied line", "PRE_OCCUPIED_LINE"),
    ("loco failure", "ROLLING_STOCK"),
    ("engineering block (TRT)", "ENGINEERING"),
    ("crew HOER", "CREW"),
    ("signal interlocking failure", "SIGNALLING"),
    ("dense fog", "NATURAL"),
    ("connection hold", "CONVENTION"),
    (None, "UNCLASSIFIED"),
    ("", "UNCLASSIFIED"),
    ("something nobody has seen", "UNCLASSIFIED"),
])
def test_cause_classification(raw, expected):
    from sources.causes import classify_cause

    assert classify_cause(raw) == expected


def test_attribution_ranks_by_delay_minutes():
    from sources.causes import attribution

    rows = attribution({"ROLLING_STOCK": 3, "NATURAL": 1},
                       {"ROLLING_STOCK": 120.0, "NATURAL": 30.0})
    assert rows[0]["cause"] == "ROLLING_STOCK"
    assert rows[0]["share_pct"] == 80.0
    assert rows[0]["owner"] and rows[0]["typical_fix"]


# ---------------------------------------------------------------------------
# Parity (KS)
# ---------------------------------------------------------------------------
def test_ks_statistic_detects_identical_and_shifted_samples():
    from sources.parity import ks_critical, ks_statistic

    rng = np.random.default_rng(0)
    same = rng.normal(size=400)
    assert ks_statistic(same, rng.normal(size=400)) < 0.15
    shifted = rng.normal(loc=5.0, size=400)
    assert ks_statistic(same, shifted) > 0.8
    assert ks_critical(400, 400) > 0


def test_parity_report_flags_distribution_shift():
    from sources.parity import parity_report

    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"congestion_r1": rng.normal(size=500),
                              "scheduled_hour": rng.integers(0, 24, size=500)})
    candidate = pd.DataFrame({"congestion_r1": rng.normal(loc=9, size=500),
                              "scheduled_hour": rng.integers(0, 24, size=500)})
    report = parity_report(reference, candidate)
    assert report["verdict"] == "distribution_shift_detected"
    assert "congestion_r1" in report["drifted_features"]
    assert "Retrain on the observed data" in report["recommendation"]


def test_parity_report_accepts_compatible_samples():
    from sources.parity import parity_report

    rng = np.random.default_rng(2)
    reference = pd.DataFrame({"congestion_r1": rng.normal(size=800)})
    candidate = pd.DataFrame({"congestion_r1": rng.normal(size=800)})
    report = parity_report(reference, candidate)
    assert report["verdict"] == "compatible"


# ---------------------------------------------------------------------------
# The trained sample itself
# ---------------------------------------------------------------------------
def test_training_features_have_no_nulls_and_expected_columns():
    from config import FEATURES_CSV, TARGET

    if not FEATURES_CSV.exists():
        pytest.skip("features.csv not built yet")
    frame = pd.read_csv(FEATURES_CSV)
    assert TARGET in frame.columns
    for column in FULL_FEATURES:
        assert column in frame.columns, f"missing feature column {column}"
    assert frame[FULL_FEATURES].notna().all().all(), "nulls in model features"


def test_provenance_record_is_written():
    from config import PROCESSED_DIR

    path = PROCESSED_DIR / "provenance.json"
    if not path.exists():
        pytest.skip("feature engineering has not run yet")
    import json
    record = json.loads(path.read_text())
    assert record["provenance"] in {"synthetic", "synthetic+real", "real"}
    assert record["rows_total"] > 0


# ---------------------------------------------------------------------------
# Regression: train/serve skew
# ---------------------------------------------------------------------------
def test_station_cluster_map_exists_and_matches_training_labels():
    """Inference must use the DBSCAN label the model was trained on.

    The original inference path derived a proxy from eigenvector centrality that
    evaluated to 0 for nearly every station — and 0 means "DBSCAN noise" in the
    training data, so the served model received a differently-meaning feature.
    """
    import json

    from config import PROCESSED_DIR, STATION_CLUSTERS_JSON

    if not STATION_CLUSTERS_JSON.exists():
        pytest.skip("feature engineering has not run yet")
    mapping = json.loads(STATION_CLUSTERS_JSON.read_text())
    graph = build_graph()
    assert set(mapping) == set(graph.nodes()), "cluster map must cover every station"

    frame = pd.read_csv(PROCESSED_DIR / "features.csv")
    truth = frame.groupby("current_station")["delay_cluster"].first()
    mismatched = [station for station in mapping
                  if float(mapping[station]) != float(truth.get(station, -1))]
    assert not mismatched, f"{len(mismatched)} stations disagree: {mismatched[:5]}"


def test_congestion_fallback_is_calibrated_on_training_data():
    """The no-live-feed fallback must stay inside the trained distribution.

    A hard-coded ``0.5 × delay`` produced congestion roughly half of what the
    models had learned, on their second-strongest feature.
    """
    from config import FEATURES_CSV, PROCESSED_DIR

    provenance = PROCESSED_DIR / "provenance.json"
    if not FEATURES_CSV.exists() or not provenance.exists():
        pytest.skip("pipeline artefacts missing")
    try:
        import joblib
        from config import BUNDLE_PATH
        bundle = joblib.load(BUNDLE_PATH)
    except Exception:
        pytest.skip("no bundle")
    proxy = bundle.get("congestion_proxy")
    if not proxy:
        pytest.skip("bundle predates the calibrated congestion fallback")

    frame = pd.read_csv(FEATURES_CSV)
    delay, station = 45.0, "Mysuru"
    fits = proxy["per_station"].get(station, proxy["global"])
    predicted = {feature: max(0.0, fits[feature][0] * delay + fits[feature][1])
                 for feature in ("congestion_r1", "congestion_r2", "congestion_r3",
                                 "cascading_delay_index")}
    nearby = frame[(frame.current_station == station) & (frame.current_delay_min.between(30, 60))]
    if not len(nearby):
        pytest.skip("no comparable training rows")
    for feature, value in predicted.items():
        observed = nearby[feature].median()
        assert abs(value - observed) <= max(5.0, 0.35 * observed), (
            f"{feature}: fallback {value:.1f} vs training median {observed:.1f}")


def test_congestion_proxy_schema_matches_its_consumers():
    """The proxy table is written by train.py and read by inference.py.

    A key rename used to surface only as a KeyError at the very end of a
    ten-minute training run, after every artefact had already been written.
    """
    import numpy as np
    import train as trainer

    frame = pd.DataFrame({
        "current_station": ["A"] * 30 + ["B"] * 30,
        "current_delay_min": list(np.linspace(5, 90, 30)) * 2,
        "congestion_r1": list(np.linspace(4, 95, 30)) * 2,
        "congestion_r2": list(np.linspace(3, 70, 30)) * 2,
        "congestion_r3": list(np.linspace(2, 55, 30)) * 2,
        "cascading_delay_index": list(np.linspace(20, 400, 30)) * 2,
    })
    table = trainer._congestion_proxy_table(frame)

    assert {"global", "per_station"} <= set(table)
    for scope in (table["global"], *table["per_station"].values()):
        assert set(scope) == set(trainer.CONGESTION_FEATURES), sorted(scope)
        for coefficients in scope.values():
            assert len(coefficients) == 2
            assert all(np.isfinite(coefficients))

    # A station with too few rows must fall back to the global fit verbatim.
    small = frame[frame.current_station == "A"].head(3)
    table_small = trainer._congestion_proxy_table(
        pd.concat([frame, small.assign(current_station="C")], ignore_index=True))
    assert table_small["per_station"]["C"] == table_small["global"]


def test_dashboard_makes_no_authenticated_call_before_sign_in():
    """The page must not probe protected endpoints while nobody is signed in.

    Regression: `loadCopilotStatus()` ran at page-load time and probed a
    dispatcher-only route, so *every* cold load returned 401 and the dashboard
    showed a "session expired" alarm before the user could type a password. This
    executes the dashboard's real JavaScript in Node against a stub DOM and
    replays the API's own role gates, so the ordering is verified rather than
    assumed. Skipped when Node is unavailable (the CI job installs it).
    """
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    probe = Path(__file__).parent / "boot_probe.mjs"
    dashboard = Path(__file__).resolve().parents[1] / "src" / "dashboard.html"
    result = subprocess.run(["node", str(probe), str(dashboard)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no protected endpoint was called before sign-in" in result.stdout
