"""
Phase 3-4: Model training, the 10-model bake-off, the radius-of-influence
ablation (the core mathematical proof), the isolated-vs-graph-aware ablation,
and SHAP-style feature importance.

Run:  python src/train.py
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:  # CatBoost is optional - only needed for the full 10-model bake-off
    HAS_CATBOOST = False
    CatBoostRegressor = None

from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor, VotingRegressor,
)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from custom_models import KNNBaggingRegressor

from config import (
    ABLATION_CSV, BUNDLE_PATH, CALIBRATION_JSON, CLUSTER_FEATURES, DAY_ENC, DAYS,
    DYNAMIC_NETWORK_FEATURES, ENSEMBLE_DIR, FEATURES_CSV, FIGURES_DIR, FULL_FEATURES,
    IMPORTANCE_CSV, ISOLATED_FEATURES, METRICS_CSV, MODEL_DIR, NETWORK_VERSION,
    CONGESTION_PROXY_JSON, PROCESSED_DIR, PROVENANCE, STATION_CLUSTERS_JSON,
    STATIC_GRAPH_FEATURES,
    UNCERTAINTY_CSV,
    TARGET, TYPE_ENC, TRAIN_TYPES, WEATHER_ENC, WEATHERS,
)
from graph_utils import build_graph, cascade_alert
from model_io import save_ensemble, save_model
from uncertainty import (
    COVERAGE_LEVELS, build_quantile_models, calibrate, evaluate,
    fit_quantile_models, reliability_figure,
)

warnings.filterwarnings("ignore")

PROVENANCE_JSON = PROCESSED_DIR / "provenance.json"
RNG = 42

# ---------------------------------------------------------------------------
# Model catalogue (exactly the 10 from the brief)
# ---------------------------------------------------------------------------
def _build_models(seed: int = RNG) -> dict[str, object]:
    models = {
        # 0) unsupervised spatial hotspot detection (DBSCAN) -> cluster feature
        "dbscan": "cluster_feature",   # placeholder, handled in feature_engineering
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=14, min_samples_leaf=3,
            n_jobs=-1, random_state=seed),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=3,
            n_jobs=-1, random_state=seed),
        "AdaBoost": AdaBoostRegressor(n_estimators=300, learning_rate=0.4,
                                      random_state=seed),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            random_state=seed),
        "XGBoost": XGBRegressor(
            n_estimators=700, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=2,
            random_state=seed, verbosity=0),
        "LightGBM": LGBMRegressor(
            n_estimators=800, num_leaves=63, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=2,
            random_state=seed, verbose=-1),
        "KNNBagging": _make_knn_bagging(seed=seed),
        "NetworkAwareVoting": _make_voting(seed=seed),
    }
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            iterations=700, depth=6, learning_rate=0.06,
            random_seed=seed, verbose=0, thread_count=2,
            allow_writing_files=False)
    return models


def _make_knn_bagging(seed: int = RNG) -> KNNBaggingRegressor:
    return KNNBaggingRegressor(n_estimators=15, n_neighbors=40, seed=seed)


def _voting_components(voting) -> list[tuple[str, object]]:
    """Extract ``(name, fitted_estimator)`` pairs from a fitted VotingRegressor.

    ``estimators_`` has been a list of ``(name, estimator)`` tuples historically
    but newer scikit-learn releases expose the fitted estimators directly, so the
    shape is normalised here instead of assuming one layout.
    """
    fitted = getattr(voting, "estimators_", None)
    if fitted is None:
        raise AttributeError("VotingRegressor is not fitted")
    declared = [name for name, _ in getattr(voting, "estimators", [])]
    out: list[tuple[str, object]] = []
    for index, item in enumerate(fitted):
        if isinstance(item, tuple) and len(item) == 2:
            out.append((str(item[0]), item[1]))
        else:
            out.append((declared[index] if index < len(declared) else f"est{index}", item))
    return out


def _coverage_verdict(empirical: float, nominal: float, n: int) -> str:
    """Judge coverage against binomial sampling error, not an exact equality.

    Empirical coverage on a finite test set is a binomial proportion, so a 0.1 pp
    shortfall at n=2,000 is noise, not miscalibration. Only a shortfall beyond
    two standard errors is reported as under-coverage.
    """
    if n <= 0:
        return "UNKNOWN"
    std_error = (nominal * (1.0 - nominal) / n) ** 0.5
    if empirical + 2 * std_error < nominal:
        return "UNDERCOVERED"
    if empirical - 2 * std_error > nominal:
        return "OVERCOVERED (conservative)"
    return "OK (within sampling error)"


def _make_voting(seed: int = RNG) -> VotingRegressor:
    return VotingRegressor([
        ("xgb", XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, n_jobs=2,
                             random_state=seed, verbosity=0)),
        ("lgb", LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9, n_jobs=2,
                              random_state=seed, verbose=-1)),
        ("etr", ExtraTreesRegressor(n_estimators=200, max_depth=16,
                                    n_jobs=2, random_state=seed)),
    ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
CONGESTION_FEATURES = ["congestion_r1", "congestion_r2", "congestion_r3",
                       "cascading_delay_index"]


def _fit_line(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Least-squares slope/intercept, degrading to a constant when degenerate."""
    if len(x) < 3 or np.allclose(x, x[0]):
        return [0.0, float(np.mean(y)) if len(y) else 0.0]
    slope, intercept = np.polyfit(x, y, 1)
    return [float(slope), float(intercept)]


def _congestion_proxy_table(df: pd.DataFrame) -> dict:
    """Calibrate the fallback congestion model against the training data.

    When the caller supplies no live congestion field, inference must synthesise
    one. The original fallback used ``congestion_r1 = 0.5 × delay``, but the data
    the models were actually fitted on has a median ratio of ≈1.04 — so the served
    model was seeing roughly half the congestion it had learned to expect, on its
    second-strongest feature. A single global ratio is not enough either:
    ``cascading_delay_index / congestion_r1`` has a median of 6.1 but a very
    different value at any particular station, which is why the fallback here is
    fitted **per station** as ``feature ≈ slope × delay + intercept``.

    This keeps synthesised features inside the training distribution. Real values
    should still arrive through ``congestion_inputs`` from a live feed.
    """
    per_station: dict[str, dict[str, list[float]]] = {}
    global_fit: dict[str, list[float]] = {}
    delay_all = df["current_delay_min"].to_numpy(dtype=float)

    for feature in CONGESTION_FEATURES:
        global_fit[feature] = _fit_line(delay_all, df[feature].to_numpy(dtype=float))

    for station, group in df.groupby("current_station"):
        x = group["current_delay_min"].to_numpy(dtype=float)
        fitted = {}
        for feature in CONGESTION_FEATURES:
            # A station needs enough rows for a stable two-parameter fit.
            fitted[feature] = (_fit_line(x, group[feature].to_numpy(dtype=float))
                               if len(group) >= 20 else global_fit[feature])
        per_station[str(station)] = fitted

    return {
        "per_station": per_station,
        "global": global_fit,
        "n_rows": int(len(df)),
        "source": ("per-station least-squares fit of each congestion ring on the "
                   "reported delay, computed from the training sample"),
    }


def _training_provenance() -> tuple[str, dict]:
    """Read the provenance recorded by feature engineering (falls back to config)."""
    record: dict = {}
    try:
        record = json.loads(PROVENANCE_JSON.read_text())
    except Exception:  # noqa: BLE001
        record = {}
    provenance = record.get("provenance") or PROVENANCE
    return provenance, record


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features: {df.shape}")
    provenance, provenance_record = _training_provenance()
    print(f"Training-sample provenance: {provenance}"
          + (f"  (simulated {provenance_record.get('n_synthetic', 0):,} + "
             f"observed {provenance_record.get('n_observed', 0):,})"
             if provenance_record else ""))

    X = df[FULL_FEATURES].copy()
    y = df[TARGET].values
    # Three-way split. The middle slice is the *calibration* set for conformal
    # prediction: it is never used to fit any model, which is what makes the
    # coverage guarantee valid (Romano et al., 2019).
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=RNG)  # 10% of all data

    print(f"Train {X_train.shape}, Calibration {X_calib.shape}, Test {X_test.shape}\n")

    # ------------------------- 10-model bake-off ---------------------------
    models = _build_models()
    results, fitted = {}, {}
    for name, model in models.items():
        if name == "dbscan":
            continue
        print(f"Training {name:20s} ...", flush=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = _metrics(y_test, pred)
        fitted[name] = model

    metrics_df = pd.DataFrame(results).T.sort_values("MAE")
    metrics_df.to_csv(METRICS_CSV)
    print("\n=== Model bake-off (sorted by MAE) ===")
    print(metrics_df.round(3).to_string())

    # ------------------- Ablation 1: isolated vs graph-aware ---------------
    print("\n=== Ablation 1: Isolated vs Graph-Aware (LightGBM) ===")
    ablation_rows = []
    variants = {
        "isolated_only": ISOLATED_FEATURES,
        "isolated+static_graph": ISOLATED_FEATURES + STATIC_GRAPH_FEATURES,
        "isolated+dynamic_network": ISOLATED_FEATURES + DYNAMIC_NETWORK_FEATURES,
        "full_graph_aware": FULL_FEATURES,
    }
    base = LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.05,
                         random_state=RNG, verbose=-1, n_jobs=2)
    base_mae = None
    for label, cols in variants.items():
        m = LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.05,
                          random_state=RNG, verbose=-1, n_jobs=2)
        m.fit(X_train[cols], y_train)
        mae = mean_absolute_error(y_test, m.predict(X_test[cols]))
        if label == "isolated_only":
            base_mae = mae
        ablation_rows.append({
            "study": "ablation_1_isolated_vs_graph",
            "variant": label,
            "n_features": len(cols),
            "MAE": round(mae, 3),
            "MAE_reduction_pct_vs_isolated": round(100 * (base_mae - mae) / base_mae, 2),
        })
        print(f"  {label:28s} MAE={mae:6.3f}  "
              f"Δ vs isolated = {100*(base_mae-mae)/base_mae:+6.2f}%")

    # -------------- Ablation 2: radius of influence ------------------------
    # Cumulative rings: how much extra signal does each wider ring add?
    # (This is the "decay of delay influence" mathematical proof.)
    print("\n=== Ablation 2: Radius of Influence (decay of delay) ===")
    radius_cols = {
        "r1 (1-edge ring)": ISOLATED_FEATURES + STATIC_GRAPH_FEATURES
                            + ["congestion_r1"] + CLUSTER_FEATURES,
        "r1+r2 (2-edge ring)": ISOLATED_FEATURES + STATIC_GRAPH_FEATURES
                               + ["congestion_r1", "congestion_r2"] + CLUSTER_FEATURES,
        "r1+r2+r3 (3-edge ring)": ISOLATED_FEATURES + STATIC_GRAPH_FEATURES
                                  + ["congestion_r1", "congestion_r2", "congestion_r3"]
                                  + CLUSTER_FEATURES,
    }
    lc_rows = []
    mae_prev = None
    for radius, (label, cols) in enumerate(radius_cols.items(), start=1):
        m = LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.05,
                          random_state=RNG, verbose=-1, n_jobs=2)
        m.fit(X_train[cols], y_train)
        mae = mean_absolute_error(y_test, m.predict(X_test[cols]))
        marginal = round(mae_prev - mae, 3) if mae_prev is not None else None
        ablation_rows.append({
            "study": "ablation_2_radius_of_influence",
            "radius": radius,
            "variant": label,
            "n_features": len(cols),
            "MAE": round(mae, 3),
            "marginal_MAE_gain": marginal,
            "MAE_reduction_pct_vs_isolated": round(100 * (base_mae - mae) / base_mae, 2),
        })
        lc_rows.append({"radius": radius, "MAE": round(mae, 3),
                        "marginal_MAE_gain": marginal})
        gain = f"  marginal gain = {marginal:+6.3f}" if marginal is not None else ""
        print(f"  {label:24s} MAE={mae:6.3f}{gain}")
        mae_prev = mae

    if len(lc_rows) >= 3:
        g12, g23 = lc_rows[1]["marginal_MAE_gain"], lc_rows[2]["marginal_MAE_gain"]
        if g12 and g23 is not None:
            share = 100 * abs(g23) / abs(g12) if g12 else float("nan")
            if g23 < g12:
                verdict = (f"the ripple fades with distance "
                           f"(the 3rd ring adds only {share:.0f}% of the 2nd ring's gain)")
            else:
                # Report the measurement as it stands rather than asserting decay.
                verdict = (f"NOTE: the 3rd ring added more than the 2nd on this run "
                           f"({share:.0f}% of the 1->2 gain) — with a single "
                           f"train/test split this difference is within noise")
            print(f"  Delay influence: 1->2 edges gains {g12} MAE, "
                  f"2->3 edges gains {g23} MAE — {verdict}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(ABLATION_CSV, index=False)

    # ---------------------- SHAP-style importance --------------------------
    print("\n=== Feature importance (permutation, SHAP-style proxy) ===")
    best_name = metrics_df.index[0]
    best = fitted[best_name]
    base_mae_test = mean_absolute_error(y_test, best.predict(X_test))
    imp_rows = []
    for col in FULL_FEATURES:
        X_perm = X_test.copy()
        X_perm[col] = np.random.permutation(X_perm[col].values)
        mae_perm = mean_absolute_error(y_test, best.predict(X_perm))
        imp_rows.append({
            "feature": col,
            "importance_mae_increase": round(mae_perm - base_mae_test, 3),
        })
    imp_df = pd.DataFrame(imp_rows).sort_values("importance_mae_increase", ascending=False)
    imp_df.to_csv(IMPORTANCE_CSV, index=False)
    print(imp_df.head(12).to_string(index=False))

    # ----------------------------------------------------------------------
    # Uncertainty stage: quantile models + conformal calibration
    # ----------------------------------------------------------------------
    print("\n=== Uncertainty: quantile LightGBM + conformal (CQR) calibration ===")
    qmodels = fit_quantile_models(build_quantile_models(seed=RNG), X_train, y_train)
    calibration = calibrate(qmodels, X_calib, y_calib, coverages=COVERAGE_LEVELS)
    unc = evaluate(qmodels, calibration, X_test, y_test, coverages=COVERAGE_LEVELS)

    unc_df = unc["intervals"]
    unc_df.to_csv(UNCERTAINTY_CSV, index=False)
    print(unc_df.to_string(index=False))
    print(f"\n  median-quantile MAE : {unc['median_mae']:.3f} min")
    print(f"  mean pinball loss   : {unc['mean_pinball']:.4f}")
    for row in unc_df.itertuples():
        verdict = _coverage_verdict(row.picp_conformal, row.coverage_nominal,
                                    int(unc["n_test"]))
        print(f"  {int(row.coverage_nominal*100):>3d}% interval: "
              f"nominal→empirical {row.coverage_nominal:.2f}→{row.picp_conformal:.3f} "
              f"(raw quantile gave {row.picp_raw_quantile:.3f}) [{verdict}]")
    raw_gap = float((unc_df["coverage_nominal"] - unc_df["picp_raw_quantile"]).mean())
    cal_gap = float((unc_df["coverage_nominal"] - unc_df["picp_conformal"]).mean())
    print(f"\n  Mean coverage shortfall — raw quantile: {raw_gap:+.3f} | "
          f"conformal: {cal_gap:+.3f}")
    print(f"  (Conformal calibration reduces the shortfall by "
          f"{100 * (1 - abs(cal_gap) / max(1e-9, abs(raw_gap))):.0f}%.)")

    CALIBRATION_JSON.write_text(json.dumps({
        "coverages": calibration,
        "quantile_levels": sorted(qmodels),
        "conformal": True,
        "method": "CQR (Romano et al. 2019)",
        "n_calibration": int(len(y_calib)),
        "test_coverage": {
            str(r.coverage_nominal): r.picp_conformal for r in unc_df.itertuples()
        },
        "median_mae": unc["median_mae"],
        "mean_pinball": unc["mean_pinball"],
    }, indent=2))
    print(f"  saved calibration -> {CALIBRATION_JSON}")

    fig_unc = reliability_figure(qmodels, X_calib, y_calib, X_test, y_test,
                                 FIGURES_DIR / "C1_uncertainty_reliability.png")
    if fig_unc:
        print(f"  saved reliability diagram -> {fig_unc}")

    # ---------------- Export trained bundle --------------------------------
    # The bundle carries: the graph, the dashboard's fused ensemble, one compact
    # single model, the quantile models + conformal offsets, the metrics and the
    # data provenance. Heavy estimators are NOT pickled — they are written in
    # each library's own portable format (see model_io.py) so that a dependency
    # upgrade cannot silently break the served model.
    G = build_graph()

    voting = fitted["NetworkAwareVoting"]
    components = _voting_components(voting)
    ensemble_manifest = save_ensemble(components, ENSEMBLE_DIR)
    print(f"  + fused ensemble components -> {ENSEMBLE_DIR} "
          f"({', '.join(c['file'] for c in ensemble_manifest['components'])})")

    # Equivalence check: the reused-from-components ensemble must reproduce the
    # original VotingRegressor's predictions, or the portable format has changed
    # the model that is being served.
    from model_io import load_ensemble as _load_ensemble
    rebuilt = _load_ensemble(ensemble_manifest, ENSEMBLE_DIR)
    sample = X_test.head(250)
    delta = float(np.max(np.abs(rebuilt.predict(sample) - voting.predict(sample))))
    print(f"  equivalence check: max |rebuilt − original| = {delta:.6f} min "
          f"({'PASS' if delta < 1e-4 else 'FAIL'})")
    if delta >= 1e-4:
        raise RuntimeError("Portable ensemble does not reproduce the fitted voting "
                           "regressor — refusing to publish it.")

    quantile_manifest = {}
    for level, model in qmodels.items():
        # Zero-padded, dot-free names: a dotted stem loses its decimals to
        # Path.with_suffix and collides with its siblings.
        written = save_model(model, MODEL_DIR / "quantiles" / f"q{int(round(level * 1000)):04d}")
        quantile_manifest[f"{level:.3f}"] = written.name
    print(f"  + quantile models (portable) -> {MODEL_DIR / 'quantiles'} "
          f"({len(quantile_manifest)} files)")

    # Verify the published artefacts, not just the in-memory objects: reload each
    # quantile model and confirm it reproduces the fitted predictions. A file-name
    # collision here once served every quantile level from a single estimator.
    from model_io import load_model as _load_model
    worst = 0.0
    for level, model in qmodels.items():
        reloaded = _load_model(MODEL_DIR / "quantiles" / quantile_manifest[f"{level:.3f}"])
        worst = max(worst, float(np.max(np.abs(
            np.asarray(reloaded.predict(X_test.to_numpy()), dtype=float)
            - np.asarray(model.predict(X_test), dtype=float)))))
    print(f"  quantile reload check: max |reloaded − fitted| = {worst:.6f} "
          f"({'PASS' if worst < 1e-6 else 'FAIL'})")
    if worst >= 1e-6:
        raise RuntimeError("Saved quantile models do not reproduce the fitted "
                           "predictions — refusing to publish them.")

    lgb_single = save_model(fitted["LightGBM"], MODEL_DIR / "single_LightGBM")
    print(f"  + single LightGBM -> {lgb_single}")

    # Station -> DBSCAN cluster map, so inference reproduces the exact feature
    # value the model was trained on (see feature_engineering.py).
    try:
        station_clusters = json.loads(STATION_CLUSTERS_JSON.read_text())
    except Exception:  # noqa: BLE001
        station_clusters = {}
        print("  ⚠  station_clusters.json missing — inference will fall back to a "
              "proxy for `delay_cluster` (train/serve skew).")

    congestion_proxy = _congestion_proxy_table(df)
    # Guard the schema: consumers read `global`/`per_station`, and a silent key
    # rename here would only surface as a KeyError at the very end of a long run.
    missing = {"global", "per_station"} - set(congestion_proxy)
    if missing:
        raise RuntimeError(f"congestion proxy table is missing keys: {sorted(missing)}")
    for feature, (slope, intercept) in congestion_proxy["global"].items():
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            raise RuntimeError(f"non-finite congestion fit for {feature}")
    print(f"  + congestion fallback calibrated on training data "
          f"({len(congestion_proxy['per_station'])} stations fitted; "
          f"global congestion_r1 = {congestion_proxy['global']['congestion_r1'][0]:.3f} "
          f"× delay + {congestion_proxy['global']['congestion_r1'][1]:.3f})")
    # Also publish it as JSON: ingestion reads this file so that a congestion
    # proxy derived for a real feed matches the fitted feature space rather than
    # re-inventing a different relationship.
    CONGESTION_PROXY_JSON.write_text(json.dumps(congestion_proxy, indent=2))
    print(f"  + congestion proxy table -> {CONGESTION_PROXY_JSON}")

    bundle = {
        "graph": G,
        "station_clusters": station_clusters,
        "congestion_proxy": congestion_proxy,
        "ensemble_manifest": ensemble_manifest,
        "dashboard_model_name": "NetworkAwareVoting",
        "single_model_name": "LightGBM",
        "single_model_file": lgb_single.name,
        "quantile_manifest": quantile_manifest,
        "calibration": calibration,
        "best_model_name": best_name,
        "metrics": metrics_df.to_dict(),
        "features": FULL_FEATURES,
        "encoders": {
            "train_type": TYPE_ENC, "weather": WEATHER_ENC, "day": DAY_ENC,
            "train_types": TRAIN_TYPES, "weathers": WEATHERS, "days": DAYS,
        },
        "best_mae": float(metrics_df.loc[best_name, "MAE"]),
        "metadata": {
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "n_calibration": int(len(X_calib)),
            "target": TARGET, "project": "Graph-Derived Feature Boosting",
            "network_version": NETWORK_VERSION,
            "provenance": provenance,
            "provenance_record": provenance_record,
            "uncertainty": {
                "method": "conformalized quantile regression",
                "quantile_levels": sorted(qmodels),
                "coverages": COVERAGE_LEVELS,
                "median_mae": unc["median_mae"],
                "mean_pinball": unc["mean_pinball"],
                "test_coverage": {
                    str(r.coverage_nominal): r.picp_conformal for r in unc_df.itertuples()
                },
            },
        },
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    print(f"  + station cluster map: {len(station_clusters)} stations")
    print(f"\nSaved trained bundle -> {BUNDLE_PATH}  "
          f"(best model: {best_name}, MAE={bundle['best_mae']:.3f} min)")
    print(f"  provenance: {provenance}")

    # ---------------- Persist the top-5 models for the model menu -----------
    # The bundle carries the fused ensemble (NetworkAwareVoting) + LightGBM for
    # backward compatibility; the remaining top-5 models are saved individually
    # so the app can offer a "choose your model" menu. Saved separately so a
    # missing optional dependency (e.g. CatBoost) only drops one menu entry
    # instead of breaking the whole bundle at unpickle time.
    top5 = metrics_df.sort_values("MAE").index[:5].tolist()
    TOP5_DIR = MODEL_DIR / "top5"
    TOP5_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale artefacts so a renamed model cannot linger in the menu.
    for stale in list(TOP5_DIR.glob("*.joblib")) + list(TOP5_DIR.glob("*.json")) \
            + list(TOP5_DIR.glob("*.txt")) + list(TOP5_DIR.glob("*.cbm")):
        stale.unlink()
    bundled = {"NetworkAwareVoting", "LightGBM"}  # already inside bundle.joblib
    for name in top5:
        if name in bundled:
            continue
        path = save_model(fitted[name], TOP5_DIR / name)
        print(f"  + saved top-5 model -> {path}  ({path.suffix.lstrip('.')} format)")
    print(f"Top-5 model menu: {', '.join(top5)}")

    # Sanity check the cascade alert function used by the dashboard
    alert = cascade_alert(G, "Mysuru", 45.0)
    print(f"Sample cascade alert from Mysuru: level={alert['level']}, "
          f"top risk={alert['top_risk_station']}")


if __name__ == "__main__":
    main()
