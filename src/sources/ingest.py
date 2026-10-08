"""
Ingestion: turn a real feed into validated, canonical observations on disk.

Flow
----
``load → map columns → normalise (graph-resolvable) → validate → append``

Observations land in ``data/raw/observed_journeys.csv`` together with a JSON
sidecar describing where they came from. ``feature_engineering.py`` merges that
file into ``features.csv`` whenever it exists, and the resulting sample
provenance (``synthetic`` / ``synthetic+real`` / ``real``) is written into the
model bundle so the app can never present simulated performance as if it were
measured on real traffic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config import RAW_DIR
from graph_utils import build_graph
from sources.base import CANONICAL_COLUMNS, map_columns, normalise_frame
from sources.validate import summarise, validate

OBSERVED_CSV = RAW_DIR / "observed_journeys.csv"
INGEST_LOG = RAW_DIR / "ingestion_log.json"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_table(path: str | Path, *, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read CSV or Excel without assuming the extension is honest."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    if suffix == ".json":
        return pd.read_json(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:               # operator exports are often cp1252
        return pd.read_csv(path, encoding="latin-1")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def ingest_dataframe(df: pd.DataFrame, *, source_name: str,
                     column_overrides: Optional[dict[str, str]] = None,
                     graph=None, max_age_days: Optional[int] = 30,
                     timestamp_column: Optional[str] = None) -> dict:
    """Validate and append a raw frame. Returns a full ingestion report."""
    graph = graph or build_graph()
    station_names = list(graph.nodes())

    mapped, mapping_report = map_columns(df, overrides=column_overrides)
    # Only the fields that cannot be reconstructed are fatal. Everything else is
    # derived from the network graph or imputed with a reported default, so a feed
    # that carries a current station, a delay and an arrival delay is usable even
    # if it looks nothing like the training CSV.
    if mapping_report["missing_required"]:
        return {
            "accepted": 0, "rejected": int(len(df)), "source": source_name,
            "mapping": mapping_report, "validation": {},
            "error": ("Required columns could not be identified: "
                      + ", ".join(mapping_report["missing_required"])
                      + ". Supply column_overrides to map them explicitly. "
                      "These three are the minimum a batch must identify: where "
                      "the train is, how late it is, and how late it reached its "
                      "destination (the label). A live feed that has no label "
                      "cannot become training data — score it with /predict/* or "
                      "/gps/* instead."),
            "hint": ("At minimum a feed must identify the current station, the "
                     "current delay and the destination arrival delay. Derivable: "
                     + ", ".join(mapping_report.get("missing_derivable", []))
                     + ". Imputed with defaults: "
                     + ", ".join(mapping_report.get("missing_imputable", []))
                     + ". Full canonical schema: " + ", ".join(CANONICAL_COLUMNS)),
        }

    canonical, normalisation = normalise_frame(mapped, graph=graph,
                                               station_names=station_names)
    valid, validation = validate(canonical, max_age_days=max_age_days,
                                 timestamp_column=timestamp_column)

    summary = summarise(validation)
    accepted = int(len(valid))
    if accepted:
        _append(valid)
        _log_ingestion(source_name, accepted, validation, mapping_report,
                       normalisation, summary)

    return {
        "accepted": accepted,
        "rejected": int(len(df) - accepted),
        "source": source_name,
        "mapping": mapping_report,
        "normalisation": normalisation,
        "validation": validation,
        "summary": summary,
        "provenance": detect_provenance(),
        "stored_at": "data/raw/observed_journeys.csv" if accepted else None,
    }


def ingest_file(path: str | Path, *, source_name: Optional[str] = None,
                sheet: Optional[str] = None, **kwargs) -> dict:
    path = Path(path)
    df = read_table(path, sheet=sheet)
    return ingest_dataframe(df, source_name=source_name or path.name, **kwargs)


def _append(rows: pd.DataFrame) -> None:
    """Append observations without ever misaligning columns.

    ``DataFrame.to_csv(mode="a")`` writes values in *its own* column order and does
    not consult the existing header. A second feed whose mapping produced a
    different order therefore wrote its values under the wrong column names —
    silently corrupting the training set. Align to the existing header, and when
    the column sets differ outright, rewrite the file with a union header.
    """
    OBSERVED_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OBSERVED_CSV.exists():
        header = pd.read_csv(OBSERVED_CSV, nrows=0).columns.tolist()
        unknown = [c for c in rows.columns if c not in header]
        missing = [c for c in header if c not in rows.columns]
        if unknown or missing:
            previous = pd.read_csv(OBSERVED_CSV)
            pd.concat([previous, rows], ignore_index=True).to_csv(OBSERVED_CSV, index=False)
            return
        rows = rows.reindex(columns=header)
    rows.to_csv(OBSERVED_CSV, mode="a", header=not OBSERVED_CSV.exists(), index=False)


def _log_ingestion(source_name: str, accepted: int, validation: dict,
                   mapping_report: dict, normalisation: dict, summary: str) -> None:
    entries: list = []
    if INGEST_LOG.exists():
        try:
            entries = json.loads(INGEST_LOG.read_text())
        except Exception:  # noqa: BLE001
            entries = []
    entries.append({
        "ts": time.time(), "source": source_name, "accepted_rows": accepted,
        "summary": summary,
        "derived_columns": normalisation.get("derived_columns", []),
        # Imputed values are assumptions, kept distinct from derived ones so the
        # log never presents a default as if it had been measured.
        "imputed_columns": normalisation.get("imputed_columns", []),
        "unusable_columns": normalisation.get("unusable_columns", []),
        "unresolved_stations": {k: v for k, v in normalisation.get("problems", {}).items()
                                if k.endswith("_unresolved")},
        "mapped_columns": mapping_report.get("mapped", {}),
        "ambiguous_columns": mapping_report.get("ambiguous_source_columns", {}),
        "missing_required": mapping_report.get("missing_required", []),
    })
    INGEST_LOG.write_text(json.dumps(entries[-200:], indent=2))


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------
def load_observed() -> Optional[pd.DataFrame]:
    """The observed observations appended so far, or ``None``."""
    if not OBSERVED_CSV.exists():
        return None
    try:
        df = pd.read_csv(OBSERVED_CSV)
    except Exception:  # noqa: BLE001
        return None
    return df if len(df) else None


def detect_provenance() -> str:
    """What the current training sample actually consists of."""
    observed = load_observed()
    n_observed = 0 if observed is None else len(observed)
    synthetic_exists = (RAW_DIR / "swr_journeys.csv").exists()
    if n_observed and synthetic_exists:
        return "synthetic+real"
    if n_observed:
        return "real"
    return "synthetic"


def ingestion_history() -> list[dict]:
    if not INGEST_LOG.exists():
        return []
    try:
        return json.loads(INGEST_LOG.read_text())
    except Exception:  # noqa: BLE001
        return []


def cause_attribution(df: Optional[pd.DataFrame] = None) -> list[dict]:
    """Delay-minutes per canonical cause head, ranked — the 'why' view."""
    from sources.causes import attribution

    df = load_observed() if df is None else df
    if df is None or "delay_cause" not in df.columns:
        return []
    counts: dict[str, int] = {}
    minutes: dict[str, float] = {}
    for _, row in df.iterrows():
        head = str(row.get("delay_cause") or "UNCLASSIFIED")
        counts[head] = counts.get(head, 0) + 1
        minutes[head] = minutes.get(head, 0.0) + float(row.get("current_delay_min") or 0.0)
    return attribution(counts, minutes)
