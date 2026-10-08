"""
Canonical journey schema and the data-source interface.

Every real-world feed says the same thing in a different dialect: one export
calls the field ``station_code``, another ``From Station``, a third ``loc``;
times arrive as ``"14:05"``, ``14`` or ``"2026-10-08T14:05:00"``. Ingesting
without normalising means the model silently trains on whatever happened to
parse — the classic way a railway ML project ends up measuring nothing.

This module defines one canonical record and the adapters that produce it, so
the rest of the pipeline only ever sees validated, typed, graph-resolvable rows.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np
import pandas as pd

from config import CONGESTION_PROXY_JSON
from station_codes import resolve_station


def congestion_proxy_table(path: Optional[Path] = None) -> dict:
    """The delay → congestion relationship published by ``train.py``.

    Inference uses this table when no live feed supplies congestion; ingestion
    uses the same one, so a real observation is expressed in the feature space the
    models were fitted on. Returns ``{}`` when the file is missing — callers then
    fall back to the documented ring proxy **and report that they did**.
    """
    global _CONGESTION_PROXY_CACHE
    if _CONGESTION_PROXY_CACHE is None:
        try:
            _CONGESTION_PROXY_CACHE = json.loads(Path(path or CONGESTION_PROXY_JSON).read_text())
        except Exception:  # noqa: BLE001 - absence is a normal state, not an error
            _CONGESTION_PROXY_CACHE = {}
    return _CONGESTION_PROXY_CACHE


_CONGESTION_PROXY_CACHE: Optional[dict] = None

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------
#: Columns the feed itself must supply. Everything else is either computed from
#: the network graph or imputed with an explicit, reported default — most real
#: operator exports carry a current station and a delay but no "next station"
#: column, and rejecting such a feed outright would make the ingestion path
#: useless in practice.
#:
#: ``current_station``                where the train is; nothing is inferable without it
#: ``current_delay_min``              the model's primary input
#: ``destination_arrival_delay_min``  the supervised label — never invented
REQUIRED_COLUMNS: list[str] = [
    "current_station", "current_delay_min", "destination_arrival_delay_min",
]

#: Columns computed from the network graph when the feed omits them.
DERIVABLE_COLUMNS: list[str] = [
    "upcoming_station", "destination", "origin", "lat", "lon",
    "remaining_km", "edge_weight_next_km",
    "congestion_r1", "congestion_r2", "congestion_r3", "cascading_delay_index",
]

#: Categorical / calendar fields that fall back to a fixed default when absent.
#: These are *imputed*, not derived from data, and are reported under a separate
#: key — calling a default train type or weather "derived" would hide an
#: assumption inside the provenance report.
IMPUTABLE_COLUMNS: list[str] = [
    "train_type", "weather", "scheduled_hour", "day_of_week",
]

#: Optional provenance/attribution columns kept alongside the model inputs.
OPTIONAL_COLUMNS: list[str] = ["train_id", "delay_cause_raw", "delay_cause", "platform"]

CANONICAL_COLUMNS: list[str] = (REQUIRED_COLUMNS + IMPUTABLE_COLUMNS
                                + DERIVABLE_COLUMNS + ["train_id"])

#: Friendly aliases seen in operator exports / open datasets.
COLUMN_SYNONYMS: dict[str, list[str]] = {
    "train_id": ["train no", "train_no", "train number", "train_number", "trainno"],
    "train_type": ["type", "train class", "service type", "class"],
    "origin": ["from", "from station", "source", "source station", "origin station"],
    "current_station": ["station", "station name", "current location", "at station",
                        "last station", "station_code", "loc", "location"],
    "upcoming_station": ["next station", "next halt", "upcoming", "to (next)"],
    "destination": ["to", "to station", "dest", "destination station"],
    "day_of_week": ["day", "dow", "weekday", "day of week"],
    "scheduled_hour": ["hour", "sch hour", "scheduled time", "dep hour", "sched hour",
                       "sch time", "sched time", "scheduled", "dep time",
                       "departure time", "scheduled departure"],
    "weather": ["wx", "weather condition"],
    "current_delay_min": ["delay", "delay min", "current delay", "delay_minutes",
                          "late by", "arrival delay"],
    "destination_arrival_delay_min": ["arrival delay", "dest delay", "final delay",
                                      "destination delay", "target"],
    "remaining_km": ["remaining distance", "dist remaining", "balance km"],
    "edge_weight_next_km": ["next section km", "section km", "next km"],
    "delay_cause_raw": ["cause", "cause code", "delay cause", "reason", "head", "cause head"],
    "platform": ["pf", "platform no", "platform number"],
}

TRAIN_TYPE_ALIASES: dict[str, str] = {
    "express": "Express", "exp": "Express", "mail": "Express", "sf exp": "Superfast",
    "sf express": "Superfast", "sup fast": "Superfast", "superfast": "Superfast",
    "sf": "Superfast", "intercity": "Intercity", "passenger": "Passenger",
    "pass": "Passenger", "memu": "MEMU", "emu": "MEMU", "freight": "Freight",
    "goods": "Freight",
}
WEATHER_ALIASES: dict[str, str] = {
    "clear": "Clear", "sunny": "Clear", "rain": "Rain", "rainy": "Rain",
    "fog": "Fog", "foggy": "Fog", "storm": "Storm", "thunder": "Storm",
}
DAYS: list[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                   "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class RailDataSource(Protocol):
    """Anything that can yield canonical journey records."""

    name: str

    def load(self) -> pd.DataFrame:  # pragma: no cover - protocol definition
        """Return a canonical DataFrame (see :data:`CANONICAL_COLUMNS`)."""
        ...


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower()).strip()


def map_columns(df: pd.DataFrame,
                overrides: Optional[dict[str, str]] = None) -> tuple[pd.DataFrame, dict]:
    """Rename incoming columns onto the canonical schema.

    ``overrides`` maps ``source column -> canonical column`` and always wins, so
    an operator with an unusual export can be supported without code changes.
    Returns the renamed frame plus a mapping report.

    A synonym claimed by two different canonical fields (``"arrival delay"``
    plausibly means either the current delay or the delay at the destination) is
    resolved deterministically — the later entry in :data:`COLUMN_SYNONYMS` wins —
    and the ambiguity is **reported** under ``ambiguous_source_columns`` so the
    operator can settle it with ``overrides`` instead of discovering it in the
    model later.
    """
    canonical = set(REQUIRED_COLUMNS + DERIVABLE_COLUMNS + IMPUTABLE_COLUMNS
                    + OPTIONAL_COLUMNS)
    lookup: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for target, aliases in COLUMN_SYNONYMS.items():
        for alias in aliases:
            key = _norm(alias)
            if key in lookup and lookup[key] != target:
                ambiguous.setdefault(key, [lookup[key]])
                if target not in ambiguous[key]:
                    ambiguous[key].append(target)
            lookup[key] = target
    # An exact canonical column name is unambiguous by definition, so it wins over
    # any alias that happens to normalise to the same string.
    for target in canonical:
        lookup[_norm(target)] = target

    renamed: dict[str, str] = {}
    used: set[str] = set()
    for column in df.columns:
        target = None
        if overrides and column in overrides:
            target = overrides[column]
        else:
            target = lookup.get(_norm(column))
        if target and target in canonical and target not in used:
            renamed[column] = target
            used.add(target)

    out = df.rename(columns=renamed)
    report = {
        "mapped": {src: dst for src, dst in renamed.items()},
        "unmapped_source_columns": [c for c in df.columns if c not in renamed],
        # Hard failure: these are the only fields that cannot be reconstructed.
        "missing_required": [c for c in REQUIRED_COLUMNS if c not in out.columns],
        # Non-fatal: computed from the network graph, or imputed with a default.
        "missing_derivable": [c for c in DERIVABLE_COLUMNS if c not in out.columns],
        "missing_imputable": [c for c in IMPUTABLE_COLUMNS if c not in out.columns],
        "ambiguous_source_columns": {c: ambiguous[_norm(c)] for c in df.columns
                                     if _norm(c) in ambiguous},
    }
    return out, report


def parse_hour(value: object) -> Optional[int]:
    """Accept ``14``, ``"14"``, ``"14:05"``, ``"2 PM"`` or a full timestamp."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) % 24 if 0 <= int(value) <= 23 else None
    text = str(value).strip().lower()
    if not text:
        return None
    time_match = re.search(r"(\d{1,2})[:.](\d{2})", text)
    if time_match:
        hour = int(time_match.group(1))
        if "pm" in text and hour < 12:
            hour += 12
        if "am" in text and hour == 12:
            hour = 0
        return hour % 24
    number = re.search(r"(\d{1,2})", text)
    if number:
        hour = int(number.group(1))
        if "pm" in text and hour < 12:
            hour += 12
        if "am" in text and hour == 12:
            hour = 0
        return hour % 24 if 0 <= hour % 24 <= 23 else None
    # ISO timestamp fallback
    try:
        return pd.Timestamp(text).hour
    except Exception:  # noqa: BLE001
        return None


def parse_day(value: object) -> Optional[int]:
    """Day of week -> ``0`` (Monday) … ``6`` (Sunday)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)) and 0 <= int(value) <= 6:
        return int(value)
    text = _norm(value)
    if not text:
        return None
    for index, day in enumerate(DAYS):
        if _norm(day).startswith(text[:3]) or text.startswith(_norm(day)):
            return index
    return None


def normalise_category(value: object, aliases: dict[str, str], allowed: Iterable[str],
                       default: str) -> str:
    text = _norm(value)
    if text in aliases:
        return aliases[text]
    for candidate in allowed:
        if _norm(candidate) == text:
            return candidate
    return default


def normalise_frame(df: pd.DataFrame, *, graph, station_names: list[str],
                    default_weather: str = "Clear",
                    default_train_type: str = "Express") -> tuple[pd.DataFrame, dict]:
    """Coerce a mapped frame into canonical, typed, graph-resolvable records.

    Congestion features (``congestion_r1`` … ``cascading_delay_index``) are
    **derived from the graph** when the feed does not carry them: the observed
    delay at the current station is projected onto its 1/2/3-edge rings with the
    same decay the simulator uses. That is a documented proxy, not a
    measurement, and it is reported as such in the ingestion report.
    """
    from graph_utils import nodes_within_radius, shortest_path_km

    out = df.copy()
    problems: dict[str, list] = {}
    derived: list[str] = []      # computed from the graph
    imputed: list[str] = []      # defaulted because the feed omitted the field
    unusable: list[str] = []     # absent and not reconstructible

    allowed_types = ["Express", "Superfast", "Intercity", "Passenger", "MEMU", "Freight"]
    allowed_weather = ["Clear", "Rain", "Fog", "Storm"]

    # --- station resolution ------------------------------------------------
    resolved: dict[str, list[str]] = {}
    for column in ("current_station", "upcoming_station", "origin", "destination"):
        if column not in out.columns:
            continue
        names = []
        unknown = []
        for value in out[column]:
            name = resolve_station(value, station_names)
            names.append(name)
            if name is None:
                unknown.append(value)
        out[column] = names
        resolved[column] = [n for n in names if n is not None]
        if unknown:
            problems[f"{column}_unresolved"] = sorted({str(u) for u in unknown})[:20]

    # A train must know where it is; destination/upcoming can default to the
    # next hop once the current station is known.
    if "current_station" in out.columns:
        out = out[out["current_station"].notna()].copy()
    if not len(out):
        raise ValueError("No rows survived station resolution — check that the feed "
                         "uses stations present in the network (see /stations).")

    if "upcoming_station" not in out.columns or out["upcoming_station"].isna().any():
        def _next_hop(row) -> Optional[str]:
            if row.get("destination") and row["destination"] != row["current_station"]:
                path = shortest_path_km(graph, row["current_station"], row["destination"])
                if path and len(path) > 1:
                    return path[1]
            neighbours = list(graph.neighbors(row["current_station"]))
            return neighbours[0] if neighbours else None
        mask = out["upcoming_station"].isna() if "upcoming_station" in out.columns \
            else pd.Series(True, index=out.index)
        out.loc[mask, "upcoming_station"] = out[mask].apply(_next_hop, axis=1)
        derived.append("upcoming_station")

    if "destination" not in out.columns:
        # No destination in the feed: fall back to the next station and say so.
        # `remaining_km` then measures the next section only, which is a weaker
        # feature than a true end-to-end distance — hence the explicit note.
        out["destination"] = out["upcoming_station"]
        derived.append("destination (assumed = next station)")
    elif out["destination"].isna().any():
        out["destination"] = out["destination"].fillna(out["upcoming_station"])
        derived.append("destination (rows missing it assumed = next station)")
    if "origin" not in out.columns:
        out["origin"] = out["current_station"]
        derived.append("origin")

    # --- types -------------------------------------------------------------
    if "train_type" in out.columns:
        out["train_type"] = [normalise_category(v, TRAIN_TYPE_ALIASES, allowed_types,
                                                default_train_type)
                             for v in out["train_type"]]
    else:
        out["train_type"] = default_train_type
        imputed.append(f"train_type (default: {default_train_type})")

    if "weather" in out.columns:
        out["weather"] = [normalise_category(v, WEATHER_ALIASES, allowed_weather,
                                             default_weather) for v in out["weather"]]
    else:
        out["weather"] = default_weather
        imputed.append(f"weather (default: {default_weather})")

    # --- time / delay ------------------------------------------------------
    if "scheduled_hour" in out.columns:
        hours = [parse_hour(v) for v in out["scheduled_hour"]]
        bad = int(sum(h is None for h in hours))
        if bad:
            problems["scheduled_hour_unparsed"] = [bad]
        out["scheduled_hour"] = [14 if h is None else h for h in hours]
    else:
        out["scheduled_hour"] = 14
        imputed.append("scheduled_hour (default: 14, i.e. afternoon)")

    if "day_of_week" in out.columns:
        days = [parse_day(v) for v in out["day_of_week"]]
        out["day_of_week"] = [0 if d is None else d for d in days]
    else:
        out["day_of_week"] = 0
        imputed.append("day_of_week (default: Monday)")

    for column in ("current_delay_min", "destination_arrival_delay_min"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            bad = int(out[column].isna().sum())
            if bad:
                problems[f"{column}_non_numeric"] = [bad]
        else:
            # Never invent a delay. Filling 0.0 here would write "arrived exactly
            # on time" — a fabricated label that passes every downstream check and
            # would train the model on invented outcomes. Leave it missing so
            # validate() rejects the rows, and record why.
            out[column] = np.nan
            unusable.append(column)
            problems[f"{column}_absent"] = ["column missing from the feed"]

    # --- distance / geo ----------------------------------------------------
    out["lat"] = out["current_station"].map(lambda s: float(graph.nodes[s]["lat"]))
    out["lon"] = out["current_station"].map(lambda s: float(graph.nodes[s]["lon"]))

    if "edge_weight_next_km" not in out.columns:
        def _edge_km(row) -> float:
            if graph.has_edge(row["current_station"], row["upcoming_station"]):
                return float(graph.edges[row["current_station"],
                                         row["upcoming_station"]]["km"])
            return 0.0
        out["edge_weight_next_km"] = out.apply(_edge_km, axis=1)
        derived.append("edge_weight_next_km")

    if "remaining_km" not in out.columns:
        def _remaining_km(row) -> float:
            path = shortest_path_km(graph, row["current_station"], row["destination"])
            if not path:
                return float(row.get("edge_weight_next_km") or 0.0)
            return float(sum(graph.edges[path[i], path[i + 1]]["km"]
                             for i in range(len(path) - 1)))
        out["remaining_km"] = out.apply(_remaining_km, axis=1)
        derived.append("remaining_km")

    # --- congestion (proxy when the feed omits it) -------------------------
    congestion_cols = ["congestion_r1", "congestion_r2", "congestion_r3",
                       "cascading_delay_index"]
    present = [c for c in congestion_cols if c in out.columns]
    if len(present) < len(congestion_cols) or out[congestion_cols].isna().any().any():
        delays = pd.to_numeric(out["current_delay_min"], errors="coerce").fillna(0.0)
        table = congestion_proxy_table()
        fits_by_station = table.get("per_station", {})
        global_fits = table.get("global", {})

        if global_fits:
            # Preferred: the same delay -> congestion relationship the models were
            # fitted on (train.py publishes it). Ingested rows then live in the
            # same feature space as the training sample instead of being projected
            # with an invented ratio that would quietly shift the feature.
            def _fit_rows() -> list[list[float]]:
                rows_out: list[list[float]] = []
                for station, delay in zip(out["current_station"], delays):
                    fits = fits_by_station.get(station) or global_fits
                    values = []
                    for column in congestion_cols:
                        slope, intercept = fits.get(column, (0.0, 0.0))
                        values.append(round(max(0.0, slope * float(delay) + intercept), 2))
                    rows_out.append(values)
                return rows_out

            projected = _fit_rows()
            source = ("CALIBRATED delay→congestion fit (models/congestion_proxy.json, "
                      "the same relationship inference uses)")
        else:
            # Last resort: the original ring-count proxy. It understates congestion
            # relative to the fitted relationship, so the report says so loudly
            # rather than presenting it as equivalent.
            cache = {}
            projected = []
            for station, delay in zip(out["current_station"], delays):
                if station not in cache:
                    cache[station] = nodes_within_radius(graph, station, radius=3)
                counts = {h: sum(1 for v in cache[station].values() if v == h)
                          for h in (1, 2, 3)}
                base = max(0.0, float(delay) * 0.5)
                r1 = base
                r2 = (r1 * counts[1] + base * 0.5 * counts[2]) / max(1, counts[1] + counts[2])
                r3 = (r1 * counts[1] + base * 0.5 * counts[2] + base * 0.25 * counts[3]) / \
                    max(1, counts[1] + counts[2] + counts[3])
                cdi = r1 * counts[1] + base * 0.5 * counts[2] + base * 0.25 * counts[3]
                projected.append([round(r1, 2), round(r2, 2), round(r3, 2), round(cdi, 2)])
            source = ("UNCALIBRATED ring proxy — models/congestion_proxy.json was not "
                      "found, so congestion is understated relative to the trained "
                      "feature space; run train.py to publish the calibration")

        for index, column in enumerate(congestion_cols):
            out[column] = [row[index] for row in projected]
        derived.append(f"congestion_r1..r3, cascading_delay_index ({source})")

    # --- cause attribution -------------------------------------------------
    if "delay_cause_raw" in out.columns:
        from sources.causes import classify_cause
        out["delay_cause"] = [classify_cause(v) for v in out["delay_cause_raw"]]
    elif "delay_cause" not in out.columns:
        out["delay_cause"] = "UNCLASSIFIED"

    if "train_id" not in out.columns:
        out["train_id"] = [f"OBS{index:06d}" for index in range(len(out))]

    report = {
        "rows_in": int(len(df)), "rows_out": int(len(out)),
        "derived_columns": derived,
        "imputed_columns": imputed,
        "unusable_columns": unusable,
        "problems": problems,
        "stations_used": len(set(out["current_station"])),
    }
    return out, report
