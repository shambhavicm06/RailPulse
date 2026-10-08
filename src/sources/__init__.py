"""
Real-data sources for RailPulse.

The published model is trained on a *simulated* delay field, because no live
NTES feed is openly available (NTES has no public API; genuine live access needs
an authorised partner or an operator's own exports). Rather than pretend
otherwise, this package provides the adapter layer that makes real data usable
the moment it is available — from a CSV/Excel export, an open dataset, or a
webhook from a live feed.

Nothing here needs the network: it normalises, validates and scores whatever
data you legitimately hold, then tells you whether it looks like the world the
model was trained on (``parity.py``).
"""
from __future__ import annotations

from sources.base import (CANONICAL_COLUMNS, COLUMN_SYNONYMS, DERIVABLE_COLUMNS,
                          REQUIRED_COLUMNS, RailDataSource, map_columns,
                          normalise_frame, parse_day, parse_hour)
from sources.causes import CAUSE_HEADS, classify_cause, describe
from sources.ingest import (cause_attribution, detect_provenance, ingest_dataframe,
                            ingest_file, ingestion_history, load_observed, read_table)
from sources.parity import ks_critical, ks_statistic, parity_figure, parity_report
from sources.validate import summarise, validate

__all__ = [
    "CANONICAL_COLUMNS", "COLUMN_SYNONYMS", "DERIVABLE_COLUMNS", "REQUIRED_COLUMNS",
    "RailDataSource", "map_columns", "normalise_frame", "parse_day", "parse_hour",
    "CAUSE_HEADS", "classify_cause", "describe",
    "cause_attribution", "detect_provenance", "ingest_dataframe", "ingest_file",
    "ingestion_history", "load_observed", "read_table",
    "ks_critical", "ks_statistic", "parity_figure", "parity_report",
    "summarise", "validate",
]
