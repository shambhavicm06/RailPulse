"""
Phase 2: Feature engineering.

Appends the NetworkX-computed STATIC graph columns (centralities) and the
DBSCAN-derived CLUSTER column to the raw journeys, then encodes categoricals.

Output: data/processed/features.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config import (
    CLUSTER_FEATURES, DAY_ENC, FEATURES_CSV, RAW_CSV, TRAIN_TYPES,
    TYPE_ENC, WEATHERS, WEATHER_ENC,
)
from graph_utils import build_graph, compute_centralities

EPS_KM = 25.0        # DBSCAN neighbourhood radius (~km) for delay-sink detection
MIN_SAMPLES = 8      # minimum delayed events to form a cluster


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


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    print(f"Raw rows: {len(df):,}")

    df = add_graph_features(df)
    print("  + static graph centrality columns (NetworkX)")

    raw_labels = _dbscan_cluster_ids(df)
    # map noise (-1) -> 0, clusters -> label + 1 (matches the inference proxy)
    df["delay_cluster"] = (raw_labels + 1)
    n_clusters = int(raw_labels.max()) + 1
    print(f"  + DBSCAN delay-sink cluster ids: "
          f"{int((raw_labels >= 0).sum())} events in {n_clusters} clusters "
          f"({int((raw_labels < 0).sum())} noise)")

    df = encode_categoricals(df)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved features -> {FEATURES_CSV}  ({len(df.columns)} columns)")


if __name__ == "__main__":
    main()
