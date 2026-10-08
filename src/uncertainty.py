"""
Uncertainty quantification for cascade-delay forecasts.

A point estimate ("48 minutes") cannot answer the question a controller actually
asks: *"will it miss the 15-minute connection?"* That is a threshold decision and
it needs a probability plus an interval with a guarantee attached.

Method: **Conformalized Quantile Regression (CQR)**
---------------------------------------------------
1. Train LightGBM quantile regressors at nine levels
   (0.025 … 0.975) — these model the conditional delay distribution, not just
   its mean.
2. On a **held-out calibration split** (never used for fitting), compute the
   conformity score ``E = max(q_lo - y, y - q_hi)`` per observation.
3. Widen every interval by the ``⌈(n+1)(1-α)⌉/n`` empirical quantile of those
   scores (Romano, Patterson & Candès, 2019).

The resulting interval carries a **finite-sample marginal coverage guarantee**:
at nominal 90 %, at least 90 % of future intervals contain the truth. Raw
quantile intervals have no such guarantee — the code reports both so the effect
of calibration is visible rather than asserted.

Reported metrics
----------------
* **PICP** — prediction-interval coverage probability (empirical coverage).
* **MPIW** — mean prediction-interval width (sharpness).
* **Pinball loss** — proper scoring rule per quantile level.

A calibrated interval must satisfy coverage ≥ nominal *and* be as narrow as
possible; both are written to ``results/uncertainty_metrics.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Quantile levels trained by the pipeline. The four supported interval levels
# (50/80/90/95 %) all map onto these without extrapolation.
QUANTILE_LEVELS: list[float] = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
COVERAGE_LEVELS: list[float] = [0.5, 0.8, 0.9, 0.95]

# How the delay distribution is summarised for the UI.
EXCEEDANCE_THRESHOLDS: list[int] = [15, 30, 60]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def build_quantile_models(seed: int = 42, n_jobs: int = 2) -> dict[float, object]:
    """One LightGBM quantile regressor per level (lazily imported)."""
    from lightgbm import LGBMRegressor

    return {
        level: LGBMRegressor(
            objective="quantile", alpha=level, n_estimators=600, num_leaves=63,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            n_jobs=n_jobs, random_state=seed, verbose=-1,
        )
        for level in QUANTILE_LEVELS
    }


def fit_quantile_models(models: dict[float, object], X_train, y_train,
                        verbose: bool = True) -> dict[float, object]:
    for level, model in models.items():
        if verbose:
            print(f"  quantile τ={level:<5} ...", flush=True)
        model.fit(X_train, y_train)
    return models


# ---------------------------------------------------------------------------
# Raw quantile predictions (monotone-rearranged)
# ---------------------------------------------------------------------------
def predict_quantiles(models: dict[float, object], X) -> np.ndarray:
    """Predict all trained quantiles, then enforce monotonicity in τ.

    Independently fitted quantile models can cross; sorting each row is the
    standard rearrangement fix and never widens an interval.
    """
    levels = sorted(models)
    preds = np.column_stack([np.asarray(models[level].predict(X), dtype=float)
                             for level in levels])
    return np.sort(preds, axis=1)


def quantile_levels(models: dict[float, object]) -> np.ndarray:
    return np.asarray(sorted(models), dtype=float)


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------
def _alpha_bounds(coverage: float) -> tuple[float, float]:
    alpha = 1.0 - coverage
    return alpha / 2.0, 1.0 - alpha / 2.0


def _pick_levels(levels: np.ndarray, lower: float, upper: float) -> tuple[int, int]:
    """Index the trained levels bracketing the requested quantiles."""
    lo = int(np.clip(np.searchsorted(levels, lower), 0, len(levels) - 1))
    hi = int(np.clip(np.searchsorted(levels, upper), 0, len(levels) - 1))
    return lo, hi


def calibrate(models: dict[float, object], X_calib, y_calib,
              coverages: list[float] | None = None) -> dict[float, dict]:
    """Compute CQR widening per coverage level on a held-out split."""
    coverages = coverages or COVERAGE_LEVELS
    levels = quantile_levels(models)
    qmat = predict_quantiles(models, X_calib)
    y = np.asarray(y_calib, dtype=float)
    n = len(y)

    out: dict[float, dict] = {}
    for coverage in coverages:
        lower_q, upper_q = _alpha_bounds(coverage)
        lo_i, hi_i = _pick_levels(levels, lower_q, upper_q)
        scores = np.maximum(qmat[:, lo_i] - y, y - qmat[:, hi_i])
        # ⌈(n+1)(1−α)⌉ / n empirical quantile of the conformity scores
        k = int(np.ceil((n + 1) * coverage))
        q_hat = float(np.quantile(scores, min(1.0, k / n), method="higher")) if n else 0.0
        out[coverage] = {
            "alpha": 1.0 - coverage,
            "lower_level": float(levels[lo_i]),
            "upper_level": float(levels[hi_i]),
            "q_hat": max(0.0, q_hat),
            "n_calibration": int(n),
        }
    return out


def intervals(models: dict[float, object], calibration: dict[float, dict], X,
              coverage: float, *, use_conformal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper bounds for one coverage level (conformal by default)."""
    specs = calibration.get(coverage)
    if specs is None:
        raise KeyError(f"coverage {coverage} was not calibrated")
    levels = quantile_levels(models)
    qmat = predict_quantiles(models, X)
    lo_i, hi_i = _pick_levels(levels, specs["lower_level"], specs["upper_level"])
    q_hat = specs["q_hat"] if use_conformal else 0.0
    lo = np.clip(qmat[:, lo_i] - q_hat, 0.0, None)
    hi = np.clip(qmat[:, hi_i] + q_hat, 0.0, None)
    return lo, hi


# ---------------------------------------------------------------------------
# Exceedance probabilities
# ---------------------------------------------------------------------------
#: A nine-point quantile estimate cannot justify claiming 0 % or 100 %; values
#: are clamped to this band and reported as *saturated* when they hit it.
PROBABILITY_BOUND = 0.005


def exceedance_probabilities(models: dict[float, object], qmat: np.ndarray,
                             thresholds: list[int] | None = None,
                             *, with_saturation: bool = False):
    """P(arrival delay > τ) per row, by inverting the predicted quantile function.

    The quantile function maps probability → delay; inverting it (delay →
    probability) gives the CDF at each threshold, and the exceedance probability
    is its complement.

    Estimates that fall outside the trained quantile range are clamped to
    ±``PROBABILITY_BOUND`` and flagged (when ``with_saturation`` is set) instead
    of being reported as a confident 0.99 for every threshold — an interval built
    from nine quantiles simply cannot support a more extreme statement.
    """
    thresholds = thresholds or EXCEEDANCE_THRESHOLDS
    probs = quantile_levels(models)
    out: dict[int, np.ndarray] = {}
    saturated: list[int] = []
    for tau in thresholds:
        # np.interp needs increasing x: delays per row are already sorted
        cdf = np.array([
            np.interp(tau, row, probs, left=0.0, right=1.0) for row in qmat
        ])
        raw = 1.0 - cdf
        if np.any((raw < PROBABILITY_BOUND) | (raw > 1.0 - PROBABILITY_BOUND)):
            saturated.append(int(tau))
        out[tau] = np.clip(raw, PROBABILITY_BOUND, 1.0 - PROBABILITY_BOUND)
    return (out, saturated) if with_saturation else out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, level: float) -> float:
    """Quantile (pinball) loss — the proper scoring rule for a quantile forecast."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(level * diff, (level - 1.0) * diff)))


def evaluate(models: dict[float, object], calibration: dict[float, dict],
             X_test, y_test, coverages: list[float] | None = None) -> dict:
    """Coverage, sharpness and pinball loss — conformal *and* raw quantile."""
    coverages = coverages or COVERAGE_LEVELS
    y = np.asarray(y_test, dtype=float)
    levels = quantile_levels(models)
    qmat = predict_quantiles(models, X_test)

    rows = []
    for coverage in coverages:
        lo_c, hi_c = intervals(models, calibration, X_test, coverage, use_conformal=True)
        lo_r, hi_r = intervals(models, calibration, X_test, coverage, use_conformal=False)
        rows.append({
            "coverage_nominal": coverage,
            "picp_conformal": round(float(np.mean((y >= lo_c) & (y <= hi_c))), 4),
            "picp_raw_quantile": round(float(np.mean((y >= lo_r) & (y <= hi_r))), 4),
            "mpiw_conformal": round(float(np.mean(hi_c - lo_c)), 3),
            "mpiw_raw_quantile": round(float(np.mean(hi_r - lo_r)), 3),
            "q_hat": round(calibration[coverage]["q_hat"], 3),
        })

    pinball = {
        str(level): round(pinball_loss(y, qmat[:, i], level), 4)
        for i, level in enumerate(levels)
    }
    median_i = int(np.argmin(np.abs(levels - 0.5)))
    return {
        "intervals": pd.DataFrame(rows),
        "pinball_loss": pinball,
        "mean_pinball": round(float(np.mean(list(pinball.values()))), 4),
        "median_mae": round(float(np.mean(np.abs(y - qmat[:, median_i]))), 4),
        "n_test": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def reliability_figure(models: dict[float, object], X_calib, y_calib,
                       X_test, y_test, out_path, *, n_sample: int = 400) -> str | None:
    """Two-panel diagnostic: coverage reliability + a real interval band.

    Panel A is the key defence — nominal vs. empirical coverage for raw quantile
    intervals and for CQR intervals, both recalibrated on the same held-out split
    so the comparison is apples-to-apples. Panel B shows the 90 % band on test
    points against the realised delays.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - figures are optional
        return None

    y = np.asarray(y_test, dtype=float)
    grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cal = calibrate(models, X_calib, y_calib, coverages=grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140)

    # Panel A — reliability curve
    nominal, emp_conf, emp_raw = [], [], []
    for cov in grid:
        nominal.append(cov)
        lo_c, hi_c = intervals(models, cal, X_test, cov, use_conformal=True)
        lo_r, hi_r = intervals(models, cal, X_test, cov, use_conformal=False)
        emp_conf.append(float(np.mean((y >= lo_c) & (y <= hi_c))))
        emp_raw.append(float(np.mean((y >= lo_r) & (y <= hi_r))))

    ax1.plot([0.4, 1.0], [0.4, 1.0], "k--", lw=1, label="perfect calibration")
    ax1.plot(nominal, emp_raw, "o-", color="#ff9800", label="raw quantile")
    ax1.plot(nominal, emp_conf, "s-", color="#1e88e5", label="conformal (CQR)")
    ax1.set_xlabel("Nominal coverage")
    ax1.set_ylabel("Empirical coverage (PICP)")
    ax1.set_title("Interval reliability — conformal vs raw quantile")
    ax1.set_xlim(0.45, 1.0)
    ax1.set_ylim(0.4, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel B — interval band on a sorted sample
    lo, hi = intervals(models, cal, X_test, 0.9, use_conformal=True)
    order = np.argsort(lo)[:n_sample]
    x = np.arange(len(order))
    ax2.fill_between(x, lo[order], hi[order], alpha=0.28, color="#1e88e5",
                     label="90 % conformal interval")
    ax2.plot(x, y[order], ".", ms=3.6, color="#111", label="actual delay")
    ax2.set_xlabel(f"Test observations (sorted by lower bound, n={len(order)})")
    ax2.set_ylabel("Destination arrival delay (min)")
    ax2.set_title("90 % prediction intervals vs. actuals")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out = str(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
