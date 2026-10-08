"""
Inference engine shared by the FastAPI backend and the Gradio dashboard.

Converts a dispatcher's live status query into graph-aware features and runs
the trained voting regressor to predict the destination arrival delay, plus a
network ripple projection for the cascade alert.
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

# Make sibling modules importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (AUTO_RETRAIN, BUNDLE_PATH, ENSEMBLE_DIR, FEATURES_CSV,
                    FULL_FEATURES, MODEL_DIR, NETWORK_VERSION, PROVENANCE,
                    PROVENANCE_LABELS, RAW_CSV)
from graph_utils import (build_graph, cascade_alert, compute_centralities,
                         nodes_within_radius, path_distance_km, shortest_path_km)
from model_io import load_ensemble, load_if_available


def ensure_bundle() -> None:
    """
    Make sure a *loadable* trained bundle exists on disk.

    Retraining policy (changed from the original silent self-heal):

    * **Missing or unreadable bundle** -> regenerate. The app cannot serve
      without a model, so this is unavoidable; it is logged loudly.
    * **Network topology changed** (``NETWORK_VERSION`` mismatch) -> retrain
      **only** when ``SWR_AUTO_RETRAIN=1``. Otherwise the change is reported and
      the existing model keeps serving, because silently swapping the live model
      mid-shift is exactly the failure mode this guard exists to prevent.
    """
    need_regen = False
    reason = ""

    if not BUNDLE_PATH.exists():
        need_regen = True
        reason = "no trained model found"
    else:
        try:
            b = joblib.load(BUNDLE_PATH)
            served = b.get("metadata", {}).get("network_version")
            if served != NETWORK_VERSION:
                if AUTO_RETRAIN:
                    need_regen = True
                    reason = (f"network topology changed "
                              f"({served} -> {NETWORK_VERSION}), SWR_AUTO_RETRAIN=1")
                else:
                    print(f"[bundle] ⚠  Network topology changed "
                          f"({served} -> {NETWORK_VERSION}) but the served model was "
                          f"trained on the older graph. Set SWR_AUTO_RETRAIN=1 (or run "
                          f"`python src/train.py`) to retrain. The existing model is "
                          f"kept in service and flagged in /system/overview.")
        except Exception as e:  # noqa: BLE001
            need_regen = True
            reason = f"{type(e).__name__}: {e}"

    if need_regen:
        print(f"[bundle] {reason} - regenerating data pipeline + retraining ...")
        from data_generator import main as gen_main
        from feature_engineering import main as fe_main
        from train import main as train_main
        gen_main()
        fe_main()
        train_main()
        print("[bundle] retraining complete.")



class CascadePredictor:
    def __init__(self, bundle_path=BUNDLE_PATH):
        ensure_bundle()
        self.bundle = joblib.load(bundle_path)
        self.G: object = self.bundle["graph"]
        self.best_name: str = self.bundle.get("best_model_name", "?")

        # ---- model menu -----------------------------------------------------
        # The fused ensemble and the compact single model are rebuilt from
        # portable per-library formats (model_io) rather than unpickled whole, so
        # a library upgrade cannot break the served model. Older bundles that
        # still carry pickled objects are accepted for backward compatibility.
        self.models: dict[str, object] = {}
        self.persistence: dict[str, str] = {}      # model name -> storage format
        self.portable: bool = False

        manifest = self.bundle.get("ensemble_manifest")
        if manifest:
            ensemble = load_ensemble(manifest, ENSEMBLE_DIR)
            if ensemble is not None:
                self.models[self.bundle.get("dashboard_model_name",
                                            "NetworkAwareVoting")] = ensemble
                self.persistence[ensemble.__class__.__name__] = "portable"
                self.portable = True
                print(f"[models] Rebuilt fused ensemble from portable components: "
                      f"{', '.join(ensemble.component_names)}")
        if "dashboard_model" in self.bundle:        # legacy pickled bundle
            self.models.setdefault(
                self.bundle.get("dashboard_model_name", "NetworkAwareVoting"),
                self.bundle["dashboard_model"])

        single_file = self.bundle.get("single_model_file")
        if single_file:
            model = load_if_available(MODEL_DIR / single_file)
            if model is not None:
                self.models[self.bundle.get("single_model_name", "LightGBM")] = model
        if "single_model" in self.bundle:            # legacy pickled bundle
            self.models.setdefault(self.bundle.get("single_model_name", "LightGBM"),
                                   self.bundle["single_model"])
        if "models" in self.bundle:  # backward-compat with a full 'models' dict
            for name, m in self.bundle["models"].items():
                self.models.setdefault(name, m)

        top5_dir = MODEL_DIR / "top5"
        if top5_dir.is_dir():
            candidates = (sorted(top5_dir.glob("*.json")) + sorted(top5_dir.glob("*.txt"))
                          + sorted(top5_dir.glob("*.cbm")) + sorted(top5_dir.glob("*.joblib")))
            for f in candidates:
                name = f.stem
                if name in self.models:
                    continue
                model = load_if_available(f)
                if model is not None:
                    self.models[name] = model
                    self.persistence[name] = "portable" if f.suffix != ".joblib" \
                        else "joblib (pinned)"

        # ---- uncertainty (quantile models + conformal offsets) --------------
        self.quantile_models: dict[float, object] = {}
        for level_key, filename in (self.bundle.get("quantile_manifest") or {}).items():
            model = load_if_available(MODEL_DIR / "quantiles" / filename)
            if model is not None:
                try:
                    self.quantile_models[round(float(level_key), 4)] = model
                except ValueError:
                    pass
        self.calibration: dict = self.bundle.get("calibration", {}) or {}
        if self.quantile_models:
            print(f"[models] Loaded {len(self.quantile_models)} quantile models "
                  f"for conformal prediction "
                  f"({len(self.calibration)} calibrated coverage levels)")


        # Default active model = the dashboard's fused ensemble.
        default = self.bundle.get("dashboard_model_name", "NetworkAwareVoting")
        self.dashboard_name = default if default in self.models else (
            self.best_name if self.best_name in self.models else next(iter(self.models)))
        self.model = self.models[self.dashboard_name]

        # Station -> DBSCAN cluster label learned at training time. Without it
        # inference would have to guess, and the guess would not match training.
        self.station_clusters: dict = self.bundle.get("station_clusters", {}) or {}
        # Empirical congestion→delay relationship, used only when the caller does
        # not supply a live congestion field.
        self.congestion_proxy: dict = self.bundle.get("congestion_proxy", {}) or {}

        self.features: list = self.bundle["features"]
        self.encoders = self.bundle["encoders"]
        self.cent = compute_centralities(self.G)

    # -- model menu ----------------------------------------------------------
    def available_models(self) -> list[dict]:
        """The selectable models with their bake-off metrics, best first."""
        metrics = self.bundle.get("metrics", {})
        # metrics is stored column-major: {"MAE": {model: val}, "RMSE": {...}, "R2": {...}}
        mae_map = metrics.get("MAE", {}) if isinstance(metrics, dict) else {}
        rmse_map = metrics.get("RMSE", {})
        r2_map = metrics.get("R2", {})

        def _num(mapping, name):
            v = mapping.get(name) if isinstance(mapping, dict) else None
            return round(float(v), 3) if v is not None else None

        out = []
        for name in self.models:
            out.append({
                "name": name,
                "mae": _num(mae_map, name),
                "rmse": _num(rmse_map, name),
                "r2": _num(r2_map, name),
            })
        out.sort(key=lambda x: (x["mae"] is None, x["mae"] if x["mae"] is not None else 1e9))

        # The menu is the "top-5 selectable models". A bundle rebuilt on a fresh
        # box (self-heal retrain) can surface a 6th candidate; keep the menu at
        # exactly 5 (best MAE first) while always retaining the active model so
        # the default prediction model stays selectable.
        if len(out) > 5:
            kept = out[:5]
            if not any(m["name"] == self.dashboard_name for m in kept):
                active = next((m for m in out if m["name"] == self.dashboard_name), None)
                if active:
                    kept[-1] = active
            out = kept
        return out

    def set_model(self, name: str) -> str:
        """Switch the active prediction model. Returns the new model name."""
        if name not in self.models:
            raise KeyError(name)
        self.model = self.models[name]
        self.dashboard_name = name
        return name

    # -- helper lookups ------------------------------------------------------
    def _cget(self, station: str, key: str, default=0.0) -> float:
        return self.cent.get(station, {}).get(key, default)

    def station_names(self) -> list[str]:
        return sorted(self.G.nodes())

    def station_meta(self, station: str) -> dict:
        lat, lon = self.G.nodes[station]["lat"], self.G.nodes[station]["lon"]
        return {"station": station, "lat": lat, "lon": lon,
                "degree": int(self.G.degree(station)),
                "eigenvector_centrality": round(self._cget(station, "eigenvector_centrality"), 4),
                "pagerank": round(self._cget(station, "pagerank"), 6)}

    # -- feature assembly ----------------------------------------------------
    def _isolated(self, train_type: str, hour: int, day: str, weather: str) -> dict:
        enc = self.encoders
        return {
            "train_type_enc": enc["train_type"].get(train_type, 0),
            "scheduled_hour": hour,
            "day_of_week": enc["day"].get(day, 0),
            "weather_enc": enc["weather"].get(weather, 0),
        }

    def _static_graph(self, current: str, upcoming: str) -> dict:
        edge_km = 0.0
        if self.G.has_edge(current, upcoming):
            edge_km = float(self.G.edges[current, upcoming]["km"])
        return {
            "edge_weight_next_km": edge_km,
            "current_station_degree": self._cget(current, "degree"),
            "current_station_degree_centrality": self._cget(current, "degree_centrality"),
            "current_station_eigenvector_centrality": self._cget(current, "eigenvector_centrality"),
            "current_station_pagerank": self._cget(current, "pagerank"),
            "current_station_betweenness_centrality": self._cget(current, "betweenness_centrality"),
            "current_station_closeness_centrality": self._cget(current, "closeness_centrality"),
            "upcoming_station_degree": self._cget(upcoming, "degree"),
            "upcoming_station_eigenvector_centrality": self._cget(upcoming, "eigenvector_centrality"),
            "upcoming_station_pagerank": self._cget(upcoming, "pagerank"),
        }

    def _dynamic(self, current: str, current_delay: float, day: str, hour: int,
                 congestion_inputs: dict | None) -> dict:
        """
        Congestion for radii 1/2/3 either comes from the user-provided live
        field (congestion_inputs) or is synthesised from the focal train's own
        delay assuming a decaying ripple around it.
        """
        day_enc = self.encoders["day"].get(day, 0)
        r1 = r2 = r3 = cdi = 0.0
        if congestion_inputs:
            # Preferred path: a live congestion field from the feed.
            r1 = float(congestion_inputs.get("congestion_r1", 0.0))
            r2 = float(congestion_inputs.get("congestion_r2", 0.0))
            r3 = float(congestion_inputs.get("congestion_r3", 0.0))
            cdi = float(congestion_inputs.get("cascading_delay_index", 0.0))
        else:
            # Fallback: synthesise the rings from the reported delay using the
            # relationship fitted on the training sample (per station where
            # possible). This keeps the served features inside the distribution
            # the models were fitted on — the previous 0.5 × delay guess put the
            # model's second-most-important feature outside its training range.
            proxy = self.congestion_proxy or {}
            fits = (proxy.get("per_station", {}).get(current)
                    or proxy.get("global", {}))
            delay = max(0.0, float(current_delay))

            def _predict(feature: str, default: float) -> float:
                coefficients = fits.get(feature)
                if not coefficients:
                    return default
                slope, intercept = coefficients
                return max(0.0, slope * delay + intercept)

            r1 = _predict("congestion_r1", delay * 0.5)
            r2 = _predict("congestion_r2", r1 * 0.78)
            r3 = _predict("congestion_r3", r1 * 0.62)
            cdi = _predict("cascading_delay_index", r1 * 6.1)
        return {
            "congestion_r1": round(r1, 2), "congestion_r2": round(r2, 2),
            "congestion_r3": round(r3, 2),
            "cascading_delay_index": round(cdi, 2),
        }

    def _cluster(self, current: str) -> float:
        """The station's DBSCAN delay-sink cluster, as seen during training.

        The label is read from the map persisted by ``feature_engineering.py``.
        The previous implementation derived a proxy from eigenvector centrality
        that evaluated to 0 for essentially every station — and 0 means "DBSCAN
        noise" in the training data, so the served model received a feature value
        with a different meaning from the one it was fitted on. The fallback is
        kept only so an older bundle still runs; it is logged as a warning.
        """
        if current in self.station_clusters:
            return float(self.station_clusters[current])
        if not getattr(self, "_cluster_warned", False):
            print("[inference] ⚠  No station→cluster map in this bundle: "
                  "`delay_cluster` falls back to a proxy and will not match "
                  "training values. Re-run src/train.py to fix.")
            self._cluster_warned = True
        ev = self._cget(current, "eigenvector_centrality")
        return float(min(4, int(ev * 10)))

    # -- public API ----------------------------------------------------------
    def predict(self, *, current_station: str, upcoming_station: str | None,
                destination: str | None, train_type: str, hour: int,
                day: str, weather: str, current_delay_min: float,
                congestion_inputs: dict | None = None,
                min_delay_for_alert: float = 0.0) -> dict:
        upcoming = upcoming_station or self._default_upcoming(current_station, destination)
        destination = destination or upcoming

        feats = {}
        feats.update(self._isolated(train_type, hour, day, weather))
        feats.update(self._static_graph(current_station, upcoming))
        feats.update(self._dynamic(current_station, current_delay_min, day, hour,
                                   congestion_inputs))
        feats["delay_cluster"] = self._cluster(current_station)

        row = pd.DataFrame([{f: feats.get(f, 0.0) for f in self.features}])[self.features]
        pred = float(np.clip(self.model.predict(row)[0], 0, None))
        pred = round(pred)

        # ripple projection + cascade alert from the dispatcher's reported delay.
        # min_delay_for_alert is 0.0 by default: a reported 0-minute delay is
        # reported and displayed as 0 — never silently bumped to a minimum.
        alert = cascade_alert(self.G, current_station,
                              max(float(current_delay_min), min_delay_for_alert))

        path = None
        if destination and destination != current_station:
            p = shortest_path_km(self.G, current_station, destination)
            path = p if p else None
        route_km = path_distance_km(self.G, path) if path else 0.0

        return {
            "predicted_destination_arrival_delay_min": pred,
            "route": path,
            "route_km": round(route_km, 1),
            "features_used": feats,
            "alert": alert,
            "model": self.dashboard_name,
            "uncertainty": self.uncertainty_report(row, float(pred)),
        }

    # -- uncertainty ---------------------------------------------------------
    def uncertainty_report(self, row: pd.DataFrame, point: float) -> dict:
        """Conformal intervals + exceedance probabilities for one feature row.

        The point estimate is forced inside every interval so the API can never
        return a headline number that contradicts its own band.
        """
        if not self.quantile_models or not self.calibration:
            return {"available": False,
                    "note": "This bundle has no quantile models. Retrain with "
                            "`python src/train.py` to enable conformal intervals."}
        from uncertainty import (EXCEEDANCE_THRESHOLDS, exceedance_probabilities,
                                 intervals, predict_quantiles, quantile_levels)

        qmat = predict_quantiles(self.quantile_models, row)
        levels = quantile_levels(self.quantile_models)
        median = float(qmat[0, int(np.argmin(np.abs(levels - 0.5)))])

        bands: dict[str, dict] = {}
        raw_contains_point = True
        for coverage in sorted(self.calibration):
            lo, hi = intervals(self.quantile_models, self.calibration, row, coverage)
            lo_f, hi_f = float(lo[0]), float(hi[0])
            if not (lo_f <= point <= hi_f):
                raw_contains_point = False
            # Guarantee the published band contains the published point estimate:
            # a headline number outside its own interval would be incoherent for a
            # controller. The unadjusted bounds are reported alongside so the
            # adjustment is visible rather than hidden.
            bands[f"{int(round(coverage * 100))}"] = {
                "lower": round(min(lo_f, point), 1),
                "upper": round(max(hi_f, point), 1),
                "width": round(max(hi_f, point) - min(lo_f, point), 1),
                "model_lower": round(lo_f, 1),
                "model_upper": round(hi_f, 1),
            }

        exc, saturated = exceedance_probabilities(self.quantile_models, qmat,
                                                 with_saturation=True)
        return {
            "available": True,
            "method": "conformalized quantile regression (CQR)",
            "median": round(median, 1),
            "intervals": bands,
            "exceedance": {f"p_gt_{t}min": round(float(v[0]), 3)
                           for t, v in exc.items()},
            "saturated_thresholds": saturated,
            "saturation_note": (
                "For these thresholds the estimate lies outside the trained "
                "quantile range; it is reported at the 0.5 % bound rather than as "
                "a falsely precise probability." if saturated else None),
            "consistency": {
                "point_inside_model_band": raw_contains_point,
                "median_minus_point_min": round(median - point, 1),
                "note": ("The point estimate comes from the served point model; the "
                         "band comes from the quantile model. They are independent "
                         "estimators of the same quantity, so the reported bounds are "
                         "expanded when necessary to contain the point. The "
                         "unadjusted bounds are given as model_lower/model_upper."
                         if not raw_contains_point else
                         "Point estimate and calibrated band agree."),
            },
            "thresholds_min": list(EXCEEDANCE_THRESHOLDS),
            "coverage_note": ("Reported bands are calibrated on a held-out split; "
                              "nominal coverage is a finite-sample guarantee, "
                              "not an estimate."),
        }

    def model_info(self) -> dict:
        """Non-secret description of what is being served (for /system/overview)."""
        meta = self.bundle.get("metadata", {})
        provenance = meta.get("provenance", PROVENANCE)
        return {
            "active_model": self.dashboard_name,
            "available_models": [m["name"] for m in self.available_models()],
            "persistence": {
                "portable_formats": self.portable,
                "note": ("XGBoost/LightGBM/CatBoost are stored in their own "
                         "version-tolerant formats; the fused ensemble is rebuilt "
                         "from components." if self.portable else
                         "Legacy pickled bundle — retrain to enable portable "
                         "persistence."),
            },
            "provenance": provenance,
            "provenance_label": PROVENANCE_LABELS.get(provenance, provenance),
            "uncertainty": {
                "enabled": bool(self.quantile_models),
                "coverage_levels": sorted(self.calibration),
                **{k: v for k, v in (meta.get("uncertainty") or {}).items()
                   if k in {"method", "median_mae", "mean_pinball", "test_coverage"}},
            },
            "network_version": meta.get("network_version"),
            "training_rows": meta.get("n_train"),
            "calibration_rows": meta.get("n_calibration"),
        }

    def _default_upcoming(self, current: str, destination: str | None) -> str:
        if destination and destination != current:
            p = shortest_path_km(self.G, current, destination)
            if p and len(p) > 1:
                return p[1]
        # highest-eigenvector neighbour
        nbrs = list(self.G.neighbors(current))
        if not nbrs:
            return current
        return max(nbrs, key=lambda n: self._cget(n, "eigenvector_centrality"))

    def model_metrics(self) -> dict:
        return self.bundle["metrics"]
