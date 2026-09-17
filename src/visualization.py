"""
Phase 1 EDA visualisations and Phase 4 result plots:
  1. The static network map (centrality-encoded nodes).
  2. The "ripple" time-lapse heatmap of a delay cascade (the storytelling plot).
  3. Model bake-off comparison bar chart.
  4. Radius-of-influence learning curve (decay of delay influence).
  5. Feature-importance bar chart (proving graph features outweigh weather).

Run:  python src/visualization.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from config import (ABLATION_CSV, FIGURES_DIR, IMPORTANCE_CSV, METRICS_CSV)
from graph_utils import build_graph, compute_centralities

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _pos(G) -> dict:
    return {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes()}


def plot_network_map(G=None, save: bool = True) -> str:
    G = G or build_graph()
    cent = compute_centralities(G)
    pos = _pos(G)
    ev = np.array([cent[n]["eigenvector_centrality"] for n in G.nodes()])
    sizes = 60 + 900 * (ev - ev.min()) / (ev.max() - ev.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#3a4a5f",
                           width=[1 + 0.06 * G.edges[e]["km"] for e in G.edges()],
                           alpha=0.6)
    sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                                node_color=ev, cmap="plasma", alpha=0.9,
                                edgecolors="black", linewidths=0.4)
    # label the top hubs only
    hubs = sorted(cent, key=lambda s: cent[s]["eigenvector_centrality"], reverse=True)[:10]
    labels = {n: n for n in hubs}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=7)
    fig.colorbar(sc, ax=ax, label="Eigenvector centrality")
    ax.set_title("South Western Railway sub-grid — hub topology (size/colour = eigenvector centrality)",
                 fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    out = FIGURES_DIR / "01_network_map.png"
    if save:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return str(out)


def plot_ripple_heatmap(G=None, source="Mysuru", delay_min=45.0, save: bool = True) -> str:
    """
    The 'story of a cascade': how a red node (a delay) turns its neighbours
    red over successive time steps. Snapshots at t=0, 1, 2, 3 propagation hops.
    """
    G = G or build_graph()
    pos = _pos(G)
    import math

    # hop distance from source
    from graph_utils import nodes_within_radius
    ring = nodes_within_radius(G, source, radius=4)
    ev = compute_centralities(G)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    t_steps = [0, 1, 2, 3]
    for ax, t in zip(axes, t_steps):
        ax.set_facecolor("#0b1220")
        node_colors, node_sizes, alphas = [], [], []
        for n in G.nodes():
            hop = ring.get(n, 99)
            if hop <= t:            # reached by the ripple
                impact = delay_min * math.exp(-0.55 * (hop - 1)) if hop >= 1 else delay_min
                impact = max(0, impact)
                node_colors.append(impact)
                node_sizes.append(120 + 6 * impact)
                alphas.append(1.0)
            else:
                node_colors.append(0)
                node_sizes.append(50)
                alphas.append(0.25)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#33465e", alpha=0.35, width=1.0)
        sc = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, cmap="inferno",
            vmin=0, vmax=max(60, delay_min), node_size=node_sizes,
            alpha=alphas, edgecolors="white", linewidths=0.4)
        labels = {n: n for n in G.nodes() if ring.get(n, 99) <= t and t > 0}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=5, font_color="white")
        ax.set_title(f"t = {t} hop(s)", color="white", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Ripple visualisation: {delay_min:.0f}-min delay at {source} propagates across the grid",
        color="white", fontsize=11)
    fig.patch.set_facecolor("#0b1220")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Projected delay impact (min)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    fig.tight_layout()
    out = FIGURES_DIR / "02_ripple_heatmap.png"
    if save:
        fig.savefig(out, bbox_inches="tight", facecolor="#0b1220")
        plt.close(fig)
    return str(out)


def plot_model_comparison(save: bool = True) -> str:
    df = pd.read_csv(METRICS_CSV, index_col=0).sort_values("MAE")
    fig, ax = plt.subplots(figsize=(10, 6))
    cols = ["#5b8ff9", "#5ad8a6", "#f6bd16", "#e8684a", "#9270ca", "#ff9d4d"]
    df["MAE"].plot(kind="barh", ax=ax, color=cols[: len(df)])
    best = df["MAE"].idxmin()
    for i, (name, mae) in enumerate(df["MAE"].items()):
        ax.text(mae + 0.1, i, f"{mae:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Mean Absolute Error (minutes)")
    ax.set_title(f"Model bake-off — best: {best}")
    ax.invert_yaxis()
    fig.tight_layout()
    out = FIGURES_DIR / "03_model_comparison.png"
    if save:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return str(out)


def plot_radius_learning_curve(save: bool = True) -> str:
    df = pd.read_csv(ABLATION_CSV)
    rdf = df[df["study"] == "ablation_2_radius_of_influence"].copy()
    rdf["radius"] = rdf["radius"].astype(int)
    rdf = rdf.sort_values("radius")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rdf["radius"], rdf["MAE"], marker="o", linewidth=2, color="#5b8ff9")
    for x, y in zip(rdf["radius"], rdf["MAE"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Cumulative congestion radius (edges)")
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("Radius of influence: each wider ring adds less information")
    ax.set_xticks([1, 2, 3])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "04_radius_learning_curve.png"
    if save:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return str(out)


def plot_feature_importance(save: bool = True) -> str:
    df = pd.read_csv(IMPORTANCE_CSV).head(15).sort_values("importance_mae_increase")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = []
    for f in df["feature"]:
        if f.startswith(("congestion", "cascading")):
            colors.append("#e8684a")            # dynamic network features -> red
        elif f in ("edge_weight_next_km",) or "centrality" in f or "degree" in f or "pagerank" in f:
            colors.append("#5b8ff9")            # static graph features -> blue
        else:
            colors.append("#9aa7b0")            # isolated features -> grey
    ax.barh(df["feature"], df["importance_mae_increase"], color=colors)
    ax.set_xlabel("MAE increase when feature is shuffled (minutes)")
    ax.set_title("Permutation feature importance — graph features dominate weather")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "05_feature_importance.png"
    if save:
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return str(out)


def main() -> None:
    print("Plotting network map ...")
    print(" ->", plot_network_map())
    print("Plotting ripple heatmap ...")
    print(" ->", plot_ripple_heatmap())
    try:
        print("Plotting model comparison ...")
        print(" ->", plot_model_comparison())
        print("Plotting radius learning curve ...")
        print(" ->", plot_radius_learning_curve())
        print("Plotting feature importance ...")
        print(" ->", plot_feature_importance())
    except FileNotFoundError as e:
        print(f"  (skip result plots — missing {e.filename}. Run train.py first)")
    print("\nDone. Figures are in results/figures/")


if __name__ == "__main__":
    main()
