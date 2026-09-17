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

from config import (BUNDLE_PATH, FEATURES_CSV, FULL_FEATURES, MODEL_DIR,
                    NETWORK_VERSION, RAW_CSV)
from graph_utils import (build_graph, cascade_alert, compute_centralities,
                         nodes_within_radius, path_distance_km, shortest_path_km)


def ensure_bundle() -> None:
    """
    Make sure a *loadable, up-to-date* trained bundle exists on disk.

    Retrains automatically when:
      - the bundle is missing or corrupted, or
      - it was pickled with different library versions, or
      - the network topology changed (NETWORK_VERSION mismatch).
    """
    need_regen = False
    reason = ""

    if not BUNDLE_PATH.exists():
        need_regen = True
        reason = "no trained model found"
    else:
        try:
            b = joblib.load(BUNDLE_PATH)
            if b.get("metadata", {}).get("network_version") != NETWORK_VERSION:
                need_regen = True
                reason = "network topology changed"
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
        # Start from the models persisted in the bundle (voting ensemble +
        # LightGBM), then merge in the individually-saved top-5 models. Loading
        # is per-file so an optional dependency (e.g. CatBoost) that is missing
        # at runtime only removes one menu entry instead of breaking the app.
        self.models: dict[str, object] = {}
        if "dashboard_model" in self.bundle:
            self.models[self.bundle.get(
                "dashboard_model_name", "NetworkAwareVoting")] = self.bundle["dashboard_model"]
        if "single_model" in self.bundle:
            self.models[self.bundle.get(
                "single_model_name", "LightGBM")] = self.bundle["single_model"]
        if "models" in self.bundle:  # backward-compat with a full 'models' dict
            for name, m in self.bundle["models"].items():
                self.models.setdefault(name, m)
        top5_dir = MODEL_DIR / "top5"
        if top5_dir.is_dir():
            for f in sorted(top5_dir.glob("*.joblib")):
                name = f.stem
                if name in self.models:
                    continue
                try:
                    self.models[name] = joblib.load(f)
                except Exception:  # noqa: BLE001 - skip unloadable model files
                    pass

        # Default active model = the dashboard's fused ensemble.
        default = self.bundle.get("dashboard_model_name", "NetworkAwareVoting")
        self.dashboard_name = default if default in self.models else (
            self.best_name if self.best_name in self.models else next(iter(self.models)))
        self.model = self.models[self.dashboard_name]

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
            r1 = float(congestion_inputs.get("congestion_r1", 0.0))
            r2 = float(congestion_inputs.get("congestion_r2", 0.0))
            r3 = float(congestion_inputs.get("congestion_r3", 0.0))
            cdi = float(congestion_inputs.get("cascading_delay_index", 0.0))
        else:
            # assume the focal delay is representative of the ring, decaying with hops
            import math
            ring = nodes_within_radius(self.G, current, radius=3)
            n1 = sum(1 for h in ring.values() if h == 1)
            n2 = sum(1 for h in ring.values() if h == 2)
            n3 = sum(1 for h in ring.values() if h == 3)
            base = max(0.0, current_delay * 0.5)
            r1 = base
            r2 = (r1 * n1 + base * 0.5 * n2) / max(1, n1 + n2)
            r3 = (r1 * n1 + base * 0.5 * n2 + base * 0.25 * n3) / max(1, n1 + n2 + n3)
            cdi = r1 * n1 + base * 0.5 * n2 + base * 0.25 * n3
        return {
            "congestion_r1": round(r1, 2), "congestion_r2": round(r2, 2),
            "congestion_r3": round(r3, 2),
            "cascading_delay_index": round(cdi, 2),
        }

    def _cluster(self, current: str) -> float:
        # nearest trained cluster label via graph-neighbour agreement; without
        # the DBSCAN artifact at inference time we use a deterministic proxy:
        # cluster id = index of the station's eigenvector-cent bucket (0..4).
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
