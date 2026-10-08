"""
Phase 2: Feature engineering.

Appends the NetworkX-computed STATIC graph columns (centralities) and the
DBSCAN-derived CLUSTER column to the raw journeys, then encodes categoricals.

Output: data/processed/features.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config import (
    CLUSTER_FEATURES, DAY_ENC, FEATURES_CSV, PROCESSED_DIR, RAW_CSV, RAW_DIR,
    STATION_CLUSTERS_JSON,
    TRAIN_TYPES, TYPE_ENC, WEATHERS, WEATHER_ENC,
)
from graph_utils import build_graph, compute_centralities

EPS_KM = 25.0        # DBSCAN neighbourhood radius (~km) for delay-sink detection
MIN_SAMPLES = 8      # minimum delayed events to form a cluster

PROVENANCE_JSON = PROCESSED_DIR / "provenance.json"
STATION_CLUSTERS_JSON = PROCESSED_DIR / "station_clusters.json"


def _dbscan_cluster_ids(df: pd.DataFrame) -> np.ndarray:
    """Cluster delayed events by lat/lon to find geographic 'delay sinks'."""
    coords = df[["lat", "lon"]].to_numpy()
    # use all events (any delay) so clusters are geographic, not delay-gated
    coords = StandardScaler().fit_transform(coords)
    eps = EPS_KM / 111.0  # ~1 degree lat ~ 111 km
    labels = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, n_jobs=-1).fit_predict(coords)
    return labels


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    G = build_graph()
    cent = compute_centralities(G)

    def cget(station: str, key: str, default: float = 0.0) -> float:
        return cent.get(station, {}).get(key, default)

    for col, key in [
        ("current_station_degree", "degree"),
        ("current_station_degree_centrality", "degree_centrality"),
        ("current_station_eigenvector_centrality", "eigenvector_centrality"),
        ("current_station_pagerank", "pagerank"),
        ("current_station_betweenness_centrality", "betweenness_centrality"),
        ("current_station_closeness_centrality", "closeness_centrality"),
    ]:
        df[col] = df["current_station"].map(lambda s: cget(s, key))

    for col, key in [
        ("upcoming_station_degree", "degree"),
        ("upcoming_station_eigenvector_centrality", "eigenvector_centrality"),
        ("upcoming_station_pagerank", "pagerank"),
    ]:
        df[col] = df["upcoming_station"].map(lambda s: cget(s, key))

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["train_type_enc"] = df["train_type"].map(TYPE_ENC)
    df["weather_enc"] = df["weather"].map(WEATHER_ENC)
    # day_of_week already numeric in the raw file
    return df


def load_training_sample() -> tuple[pd.DataFrame, dict]:
    """Synthetic journeys plus any real observations that have been ingested.

    Returns the combined frame and a provenance record. When
    ``data/raw/observed_journeys.csv`` exists (written by
    ``sources/ingest.py``) its rows are merged in, and the provenance is
    upgraded accordingly so the app reports what the model was really trained
    on rather than assuming the simulator.
    """
    frames = []
    composition = {"n_synthetic": 0, "n_observed": 0, "sources": []}

    if RAW_CSV.exists():
        synthetic = pd.read_csv(RAW_CSV)
        composition["n_synthetic"] = int(len(synthetic))
        composition["sources"].append({"name": "simulated delay field",
                                       "rows": int(len(synthetic))})
        frames.append(synthetic)

    observed_path = RAW_DIR / "observed_journeys.csv"
    if observed_path.exists():
        observed = pd.read_csv(observed_path)
        if len(observed):
            composition["n_observed"] = int(len(observed))
            composition["sources"].append({"name": "ingested real observations",
                                           "rows": int(len(observed))})
            frames.append(observed)

    if not frames:
        raise FileNotFoundError(
            "No raw data found. Run `python src/data_generator.py` first, or "
            "ingest observations with the /ingest endpoints.")

    df = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]

    if composition["n_observed"] and composition["n_synthetic"]:
        composition["provenance"] = "synthetic+real"
    elif composition["n_observed"]:
        composition["provenance"] = "real"
    else:
        composition["provenance"] = "synthetic"
    composition["rows_total"] = int(len(df))
    return df, composition


def main() -> None:
    df, composition = load_training_sample()
    print(f"Raw rows: {len(df):,}  "
          f"(simulated {composition['n_synthetic']:,} + observed "
          f"{composition['n_observed']:,})")
    print(f"  provenance: {composition['provenance']}")

    df = add_graph_features(df)
    print("  + static graph centrality columns (NetworkX)")

    raw_labels = _dbscan_cluster_ids(df)
    # map noise (-1) -> 0, clusters -> label + 1
    df["delay_cluster"] = (raw_labels + 1)
    n_clusters = int(raw_labels.max()) + 1
    print(f"  + DBSCAN delay-sink cluster ids: "
          f"{int((raw_labels >= 0).sum())} events in {n_clusters} clusters "
          f"({int((raw_labels < 0).sum())} noise)")

    # Persist the per-station cluster assignment.
    #
    # WHY: DBSCAN labels are only meaningful at training time. The original
    # inference path recomputed a *proxy* from eigenvector centrality, which
    # almost always evaluated to 0 — and 0 in training means "DBSCAN noise".
    # The served model was therefore fed a feature value that meant something
    # different from what it learned: classic train/serve skew. A station's
    # coordinates are fixed, so one deterministic label per station reproduces
    # the training-time value exactly.
    station_clusters = (
        df.groupby("current_station")["delay_cluster"]
          .agg(lambda values: int(values.mode().iloc[0]))
          .to_dict()
    )
    STATION_CLUSTERS_JSON.write_text(json.dumps(station_clusters, indent=2, sort_keys=True))
    print(f"  + per-station cluster map -> {STATION_CLUSTERS_JSON} "
          f"({len(station_clusters)} stations, ids 0–{int(df['delay_cluster'].max())})")

    df = encode_categoricals(df)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved features -> {FEATURES_CSV}  ({len(df.columns)} columns)")

    # Record what this sample actually contains. train.py copies it into the
    # model bundle and the dashboard shows it as a badge, so a reviewer can see
    # at a glance whether the metrics came from simulated or observed traffic.
    composition["features_csv"] = str(FEATURES_CSV.name)
    composition["n_features"] = int(len(df.columns))
    provenances = [s for s in df.get("provenance", [])] if "provenance" in df else []
    PROVENANCE_JSON.write_text(json.dumps(composition, indent=2))
    print(f"  provenance record -> {PROVENANCE_JSON}  "
          f"({composition['provenance']})")
    if provenances:
        print(f"  per-row provenance values: {sorted(set(provenances))}")


if __name__ == "__main__":
    main()
