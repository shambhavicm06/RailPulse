"""
Graph utilities: build the SWR rail network, extract network-mathematics
features (the "Graph-to-Tabular flattening"), and compute radius-based
congestion / ripple helpers.
"""
from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
import numpy as np

from config import EDGES, STATIONS


def build_graph() -> nx.Graph:
    """Construct the undirected rail grid: stations = nodes, tracks = edges."""
    G = nx.Graph()
    for name, (lat, lon) in STATIONS.items():
        G.add_node(name, lat=lat, lon=lon)
    for a, b, km in EDGES:
        G.add_edge(a, b, km=km)
    return G


def compute_centralities(G: nx.Graph) -> dict[str, dict[str, float]]:
    """
    Compute the full set of network-mathematics features per station.
    These are the 1D columns we 'flatten' the 2D graph into.
    """
    dc = nx.degree_centrality(G)
    ev = _eigenvector(G)
    pr = nx.pagerank(G, weight="km")
    bc = nx.betweenness_centrality(G, weight="km")
    cc = nx.closeness_centrality(G)
    out: dict[str, dict[str, float]] = {}
    for s in G.nodes():
        out[s] = {
            "degree": int(G.degree(s)),
            "degree_centrality": dc[s],
            "eigenvector_centrality": ev[s],
            "pagerank": pr[s],
            "betweenness_centrality": bc[s],
            "closeness_centrality": cc[s],
        }
    return out


def _eigenvector(G: nx.Graph) -> dict[str, float]:
    try:
        return nx.eigenvector_centrality(G, weight="km", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        # fall back to a damping-based variant on non-convergence
        return nx.eigenvector_centrality_numpy(G, weight="km")


def shortest_path_km(G: nx.Graph, origin: str, destination: str) -> list[str] | None:
    try:
        return nx.shortest_path(G, source=origin, target=destination, weight="km")
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return None


def path_distance_km(G: nx.Graph, path: Iterable[str]) -> float:
    path = list(path)
    return float(sum(G[path[i]][path[i + 1]]["km"] for i in range(len(path) - 1)))


def nodes_within_radius(G: nx.Graph, station: str, radius: int) -> dict[str, int]:
    """Map of node -> hop-distance for all nodes within `radius` edges."""
    seen = {station: 0}
    frontier = [station]
    hops = 0
    while frontier and hops < radius:
        hops += 1
        nxt = []
        for n in frontier:
            for nb in G.neighbors(n):
                if nb not in seen:
                    seen[nb] = hops
                    nxt.append(nb)
        frontier = nxt
    return seen


def congestion_at(G: nx.Graph, delay_field, station: str, day: int, hour: int,
                  radius: int, mode: str = "mean", decay: float = 0.5) -> float:
    """
    Aggregate the live delay field over all *other* trains occupying stations
    within `radius` edges of `station`.

    mode == "mean"  -> Grid Congestion Score (average delay).
    mode == "sum"   -> Cascading Delay Index (aggregated minutes, hop-decayed).
    """
    ring = nodes_within_radius(G, station, radius)
    vals = []
    for n, hop in ring.items():
        if n == station:           # "other trains" => exclude the focal station
            continue
        vals.append(delay_field.get((n, day, hour), 0.0) * math.exp(-decay * (hop - 1)))
    if not vals:
        return 0.0
    return float(np.mean(vals)) if mode == "mean" else float(np.sum(vals))


def ripple_projection(G: nx.Graph, source: str, delay_min: float,
                      radius: int = 3, decay: float = 0.55) -> list[dict]:
    """
    Project how `delay_min` at `source` cascades to downstream stations.
    Returns a sorted list of {station, hop, projected_impact, eigenvector}.
    """
    centralities = compute_centralities(G)
    ring = nodes_within_radius(G, source, radius)
    rows = []
    for n, hop in ring.items():
        if n == source:
            continue
        ev = centralities[n]["eigenvector_centrality"]
        impact = delay_min * math.exp(-decay * (hop - 1)) * (1.0 + ev)
        rows.append({
            "station": n,
            "hop": hop,
            "projected_impact_min": round(impact, 1),
            "eigenvector_centrality": round(ev, 4),
        })
    rows.sort(key=lambda r: r["projected_impact_min"], reverse=True)
    return rows


def cascade_alert(G: nx.Graph, source: str, delay_min: float,
                  radius: int = 3) -> dict:
    """
    Produce the dispatcher-facing "Grid Cascade Alert" summary.
    """
    rows = ripple_projection(G, source, delay_min, radius)
    at_risk = [r for r in rows if r["projected_impact_min"] >= 15.0]
    if delay_min >= 45:
        level, tag = "CRITICAL", "SEVERE"
    elif delay_min >= 25:
        level, tag = "HIGH", "HIGH"
    elif delay_min >= 10:
        level, tag = "MEDIUM", "ELEVATED"
    else:
        level, tag = "LOW", "NOMINAL"

    return {
        "level": level,
        "tag": tag,
        "source": source,
        "source_delay_min": round(float(delay_min), 1),
        "at_risk_stations": at_risk[:8],
        "top_risk_station": at_risk[0]["station"] if at_risk else None,
    }
