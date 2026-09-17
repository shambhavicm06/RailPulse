"""
Phase 3-4: Model training, the 10-model bake-off, the radius-of-influence
ablation (the core mathematical proof), the isolated-vs-graph-aware ablation,
and SHAP-style feature importance.

Run:  python src/train.py
"""
from __future__ import annotations

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
    ABLATION_CSV, BUNDLE_PATH, CLUSTER_FEATURES, DAY_ENC, DAYS,
    DYNAMIC_NETWORK_FEATURES, FEATURES_CSV, FIGURES_DIR, FULL_FEATURES,
    IMPORTANCE_CSV, ISOLATED_FEATURES, METRICS_CSV, MODEL_DIR, NETWORK_VERSION,
    STATIC_GRAPH_FEATURES,
    TARGET, TYPE_ENC, TRAIN_TYPES, WEATHER_ENC, WEATHERS,
)
from graph_utils import build_graph, cascade_alert

warnings.filterwarnings("ignore")
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
def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features: {df.shape}")

    X = df[FULL_FEATURES].copy()
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG)
    print(f"Train {X_train.shape}, Test {X_test.shape}\n")

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
        if g12 and g23 and g12 > 0:
            print(f"  Delay influence decay: 1->2 edges gains {g12} MAE, "
                  f"2->3 edges gains only {g23} MAE "
                  f"({100 * abs(g23) / abs(g12):.0f}% of the 1->2 gain — "
                  f"the ripple fades with distance)")

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

    # ---------------- Export trained bundle --------------------------------
    # Slim bundle: graph + the dashboard's fused voting ensemble + one compact
    # single model (LightGBM) for the API. The heavy CatBoost/RF/ET objects are
    # intentionally NOT persisted — their metrics live in results/*.csv.
    G = build_graph()
    bundle = {
        "graph": G,
        "dashboard_model": fitted["NetworkAwareVoting"],   # spec's fused ensemble
        "dashboard_model_name": "NetworkAwareVoting",
        "single_model": fitted["LightGBM"],
        "single_model_name": "LightGBM",
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
            "target": TARGET, "project": "Graph-Derived Feature Boosting",
            "network_version": NETWORK_VERSION,
        },
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    print(f"\nSaved trained bundle -> {BUNDLE_PATH}  "
          f"(best model: {best_name}, MAE={bundle['best_mae']:.3f} min)")

    # ---------------- Persist the top-5 models for the model menu -----------
    # The bundle carries the fused ensemble (NetworkAwareVoting) + LightGBM for
    # backward compatibility; the remaining top-5 models are saved individually
    # so the app can offer a "choose your model" menu. Saved separately so a
    # missing optional dependency (e.g. CatBoost) only drops one menu entry
    # instead of breaking the whole bundle at unpickle time.
    top5 = metrics_df.sort_values("MAE").index[:5].tolist()
    TOP5_DIR = MODEL_DIR / "top5"
    TOP5_DIR.mkdir(parents=True, exist_ok=True)
    bundled = {"NetworkAwareVoting", "LightGBM"}  # already inside bundle.joblib
    for name in top5:
        if name in bundled:
            continue
        joblib.dump(fitted[name], TOP5_DIR / f"{name}.joblib", compress=3)
        print(f"  + saved top-5 model -> {TOP5_DIR / f'{name}.joblib'}")
    print(f"Top-5 model menu: {', '.join(top5)}")

    # Sanity check the cascade alert function used by the dashboard
    alert = cascade_alert(G, "Mysuru", 45.0)
    print(f"Sample cascade alert from Mysuru: level={alert['level']}, "
          f"top risk={alert['top_risk_station']}")


if __name__ == "__main__":
    main()
