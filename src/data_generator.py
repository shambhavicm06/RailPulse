"""
Phase 1 synthetic-data engine.

Live NTES scraping is not possible in a sandbox, so we generate a realistic,
*network-coupled* dataset for the South Western Railway sub-grid:

1. A spatio-temporal DELAY FIELD is simulated on the graph: delays are seeded
   per station/hour and then PROPAGATED along the edges with exponential decay
   (this is the physics of the "cascading ripple effect").
2. 10,000 journeys are sampled: each journey reads the live congestion in the
   rings around its current station, so graph structure genuinely drives the
   target (destination arrival delay).

Output: data/raw/swr_journeys.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DAY_ENC, DAYS, DELAY_SINK_BOOST, RAW_CSV, STATIONS,
    TYPE_PENALTY, TRAIN_TYPES, WEATHERS, WEATHER_PENALTY, peak_load, peak_penalty,
)
from graph_utils import build_graph, congestion_at, nodes_within_radius, path_distance_km, shortest_path_km

N_DAYS = 30
N_JOURNEYS = 10_000


def simulate_delay_field(G, n_days: int, seed: int = 42) -> dict:
    """
    Simulate a delay field: field[(station, day, hour)] -> delay in minutes.
    Delays propagate along the graph edges with exponential distance decay,
    so neighbouring stations become correlated (the ripple).
    """
    rng = np.random.default_rng(seed)
    stations = list(STATIONS.keys())

    # station-level delay propensity (some junctions are chronic delay sinks)
    sink = {s: float(rng.uniform(0.4, 1.0)) for s in stations}
    for s, boost in DELAY_SINK_BOOST.items():
        sink[s] = min(2.0, sink[s] * boost)

    field: dict[tuple[str, int, int], float] = {}
    for d in range(n_days):
        for h in range(24):
            load = peak_load(h)
            seed_delay = {
                s: float(rng.gamma(shape=1.6, scale=2.0) * sink[s] * load)
                for s in stations
            }
            # random incidents: a big local disruption somewhere
            for _ in range(int(rng.integers(0, 3))):
                s = stations[int(rng.integers(0, len(stations)))]
                seed_delay[s] += float(rng.uniform(20, 60))

            cur = dict(seed_delay)
            for _ in range(3):  # 3 propagation rounds = up to 3-edge ripple
                nxt = dict(cur)
                for s in stations:
                    inflow = sum(
                        cur[n] * float(np.exp(-G.edges[s, n]["km"] / 50.0))
                        for n in G.neighbors(s)
                    )
                    nxt[s] = cur[s] + 0.30 * inflow
                cur = nxt

            for s in stations:
                field[(s, d, h)] = cur[s]
    return field


def generate_journeys(G, field, n_journeys: int = N_JOURNEYS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stations = list(STATIONS.keys())
    rows = []

    for i in range(n_journeys):
        origin = stations[int(rng.integers(0, len(stations)))]
        dest = stations[int(rng.integers(0, len(stations)))]
        if dest == origin:
            dest = stations[(stations.index(origin) + 1) % len(stations)]

        path = shortest_path_km(G, origin, dest)
        if path is None or len(path) < 2:
            continue
        # position the train somewhere along its route (not at the terminus)
        idx = int(rng.integers(0, len(path) - 1))
        current = path[idx]
        upcoming = path[idx + 1] if idx + 1 < len(path) else current

        day = int(rng.integers(0, N_DAYS))
        hour = int(rng.integers(0, 24))
        train_type = TRAIN_TYPES[int(rng.integers(0, len(TRAIN_TYPES)))]
        weather = WEATHERS[int(rng.integers(0, len(WEATHERS)))]

        remaining_km = path_distance_km(G, path[idx:])
        edge_km = float(G.edges[current, upcoming]["km"]) if upcoming != current else 0.0

        # the train's *own* accumulated delay at the current station is driven
        # by the live field there
        current_delay = float(field.get((current, day, hour), 0.0) + rng.normal(0, 2.0))
        current_delay = max(0.0, current_delay)

        # dynamic network features as observed by the dispatcher
        congestion_r1 = congestion_at(G, field, current, day, hour, radius=1, mode="mean")
        congestion_r2 = congestion_at(G, field, current, day, hour, radius=2, mode="mean")
        congestion_r3 = congestion_at(G, field, current, day, hour, radius=3, mode="mean")
        cascading_delay_index = congestion_at(G, field, current, day, hour, radius=3, mode="sum")

        # congestion *ahead* along the remaining path (drives the target, so the
        # graph structure genuinely matters for the label)
        ahead = [congestion_at(G, field, s, day, hour, radius=2, mode="mean")
                 for s in path[idx + 1: idx + 6]]
        path_congestion = float(np.mean(ahead)) if ahead else congestion_r2

        # ---- target: destination arrival delay (minutes) -------------------
        km_eff = min(remaining_km, 1000.0)   # cap for cross-country journeys
        delay = (
            current_delay
            + 0.55 * path_congestion * (km_eff / 100.0) ** 0.55
            + 0.35 * cascading_delay_index / 5.0
            + WEATHER_PENALTY[weather]
            + TYPE_PENALTY[train_type]
            + peak_penalty(hour)
            + (2.0 if day % 7 >= 5 else 0.0)
            + float(rng.normal(0, 3.0))
        )
        delay = max(0, min(300, round(delay)))

        lat, lon = STATIONS[current]
        rows.append({
            "train_id": f"TRN{i:05d}",
            "train_type": train_type,
            "day_of_week": DAY_ENC[DAYS[day % 7]],
            "scheduled_hour": hour,
            "weather": weather,
            "origin": origin,
            "current_station": current,
            "upcoming_station": upcoming,
            "destination": dest,
            "current_delay_min": round(current_delay, 1),
            "remaining_km": round(remaining_km, 1),
            "edge_weight_next_km": edge_km,
            "lat": lat,
            "lon": lon,
            "congestion_r1": round(congestion_r1, 2),
            "congestion_r2": round(congestion_r2, 2),
            "congestion_r3": round(congestion_r3, 2),
            "cascading_delay_index": round(cascading_delay_index, 2),
            "destination_arrival_delay_min": delay,
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("Building SWR rail graph ...")
    G = build_graph()
    print(f"  nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    print("Simulating spatio-temporal delay field (30 days x 24h) ...")
    field = simulate_delay_field(G, N_DAYS)

    print(f"Generating {N_JOURNEYS:,} journeys ...")
    df = generate_journeys(G, field, N_JOURNEYS)
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(df):,} rows -> {RAW_CSV}")
    print("Target stats (minutes):")
    print(df["destination_arrival_delay_min"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
