"""
Synthetic vs. real parity testing.

The published model is trained on simulated data. The moment real observations
arrive, the honest question is not *"does the model still score well?"* but
*"is the real data even drawn from the world the model was trained on?"*. If the
distributions have drifted, an accuracy figure computed on synthetic data tells
you nothing about performance on the real network, and the app should say so
instead of quietly retraining.

Method: a two-sample **Kolmogorov–Smirnov** statistic per feature, computed here
in ~15 lines rather than importing scipy (the project's dependency list stays
minimal, and the statistic is easy to audit).

Interpretation follows the standard KS critical value ``c(α) = sqrt(-ln(α/2) /
2) · sqrt((n₁+n₂)/(n₁n₂))``: with the default α = 0.01 a feature is flagged when
its statistic exceeds that threshold, i.e. when the two samples are inconsistent
at the 1 % level.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PARITY_ALPHA = 0.01


def ks_statistic(x: Iterable[float], y: Iterable[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max ECDF gap)."""
    a = np.sort(np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float))
    b = np.sort(np.asarray([v for v in y if v is not None and np.isfinite(v)], dtype=float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_critical(n_a: int, n_b: int, alpha: float = PARITY_ALPHA) -> float:
    """KS critical value for the given sample sizes and significance level."""
    if n_a <= 0 or n_b <= 0:
        return float("inf")
    return math.sqrt(-math.log(alpha / 2.0) / 2.0) * math.sqrt((n_a + n_b) / (n_a * n_b))


def parity_report(reference: pd.DataFrame, candidate: pd.DataFrame,
                  features: Optional[list[str]] = None,
                  alpha: float = PARITY_ALPHA) -> dict:
    """Compare two feature frames column-wise.

    ``reference`` is what the model was trained on; ``candidate`` is the new
    (typically real) data. Returns per-feature KS statistics plus an overall
    verdict that the dashboard surfaces as a data-drift warning.
    """
    if features is None:
        features = [c for c in reference.columns
                    if c in candidate.columns
                    and pd.api.types.is_numeric_dtype(reference[c])]

    rows = []
    for column in features:
        stat = ks_statistic(reference[column].to_numpy(), candidate[column].to_numpy())
        crit = ks_critical(len(reference), len(candidate), alpha)
        drifted = bool(np.isfinite(stat) and stat > crit)
        rows.append({
            "feature": column,
            "ks_statistic": round(stat, 4) if np.isfinite(stat) else None,
            "critical_value": round(crit, 4) if np.isfinite(crit) else None,
            "drifted": drifted,
            "train_mean": round(float(np.nanmean(reference[column])), 3),
            "new_mean": round(float(np.nanmean(candidate[column])), 3),
        })

    drifted_rows = [r for r in rows if r["drifted"]]
    table = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False,
                                           na_position="last")
    n_compared = len([r for r in rows if r["ks_statistic"] is not None])
    return {
        "alpha": alpha,
        "n_reference": int(len(reference)),
        "n_candidate": int(len(candidate)),
        "features_compared": n_compared,
        "features_drifted": len(drifted_rows),
        "max_ks": round(float(table["ks_statistic"].max()), 4) if n_compared else None,
        "verdict": ("compatible" if not drifted_rows else "distribution_shift_detected"),
        "recommendation": (
            "Real data is consistent with the training distribution — retraining "
            "can be considered on the usual schedule."
            if not drifted_rows else
            "Distribution shift detected: the model was trained on a different "
            "world than this data. Retrain on the observed data and report metrics "
            "from it before quoting any accuracy figure."
        ),
        "table": table,
        "drifted_features": [r["feature"] for r in drifted_rows],
    }


def parity_figure(report: dict, out_path, *, top_n: int = 12) -> Optional[str]:
    """Horizontal bar chart of per-feature KS statistics against the threshold."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    table = report.get("table")
    if table is None or not len(table):
        return None
    table = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.4, 0.42 * len(table))), dpi=140)
    colors = ["#d32f2f" if d else "#1e88e5" for d in table["drifted"]]
    ax.barh(table["feature"], table["ks_statistic"], color=colors)
    if report.get("features_compared"):
        crit = table["critical_value"].dropna()
        if len(crit):
            ax.axvline(float(crit.iloc[0]), color="#333", ls="--", lw=1.2,
                       label=f"KS threshold (α={report.get('alpha', 0.01)})")
            ax.legend(fontsize=9)
    ax.set_xlabel("KS statistic (synthetic reference vs. new data)")
    ax.set_title("Data-parity check — red bars indicate distribution shift")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def compare_feature_files(reference_csv: str | Path, candidate_csv: str | Path,
                          features: Optional[list[str]] = None) -> dict:
    """Convenience wrapper for the CLI: compare two ``features.csv`` files."""
    reference = pd.read_csv(reference_csv)
    candidate = pd.read_csv(candidate_csv)
    return parity_report(reference, candidate, features=features)
