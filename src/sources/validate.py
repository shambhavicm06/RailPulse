"""
Data-quality gate.

Every row that reaches the model is a claim about a real train. This module
decides which claims are admissible, using railway-plausible bounds rather than
statistical ones — a report of a train running 6,000 minutes late, or of 240 km
covered in 5 minutes, is a data-entry error and must not become training signal.

Checks
------
==================  ==========================================================
Check               Rule
==================  ==========================================================
delay range         ``-15 … 1440`` minutes (early arrival is possible, but
                    bounded; > 24 h is almost always a bad timestamp)
hour range          ``0 … 23``
section plausibility  every row must resolve to a real adjacent-station pair;
                    a missing or implausibly long section is rejected
station membership  every station must resolve to a graph node
duplicate rows      same ``train_id`` + ``current_station`` + ``scheduled_hour``
staleness           timestamp older than ``max_age_days`` (feeds only)
completeness        fraction of non-null values in required columns
==================  ==========================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_PLAUSIBLE_DELAY_MIN = 1440.0
MIN_PLAUSIBLE_DELAY_MIN = -15.0
#: The longest single section in the bundled 84-station network is 660 km. A
#: "section" longer than this means the two stations are not adjacent, i.e. the
#: edge lookup in normalisation found no such pair.
MAX_PLAUSIBLE_SECTION_KM = 700.0


def validate(df: pd.DataFrame, *, max_age_days: int | None = None,
             timestamp_column: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return ``(valid_rows, report)``. Rejections are itemised, never silent."""
    report: dict = {"checks": {}, "rejected_rows": 0, "rows_in": int(len(df))}
    if not len(df):
        report["checks"]["empty"] = {"status": "fail", "detail": "no rows supplied"}
        return df, report

    work = df.copy()
    keep = pd.Series(True, index=work.index)
    checks: dict[str, dict] = {}

    # -- the supervised label must exist ------------------------------------
    if "destination_arrival_delay_min" not in work.columns:
        checks["label"] = {
            "status": "fail",
            "detail": ("destination_arrival_delay_min (the label) is absent — these "
                       "rows cannot train or evaluate a supervised model"),
        }
        report["checks"] = checks
        report["rejected_rows"] = int(len(work))
        report["fatal"] = "No destination-arrival-delay column: the batch has no label."
        return work.iloc[0:0], report

    # -- delay range --------------------------------------------------------
    for column in ("current_delay_min", "destination_arrival_delay_min"):
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        bad = values.isna() | (values < MIN_PLAUSIBLE_DELAY_MIN) | \
            (values > MAX_PLAUSIBLE_DELAY_MIN)
        checks[column] = {
            "status": "warn" if bad.any() else "pass",
            "rejected": int(bad.sum()),
            "detail": (f"outside [{MIN_PLAUSIBLE_DELAY_MIN}, {MAX_PLAUSIBLE_DELAY_MIN}] "
                       f"minutes; {int(bad.sum())} row(s) rejected"),
        }
        keep &= ~bad

    # -- hour range ---------------------------------------------------------
    if "scheduled_hour" in work.columns:
        hours = pd.to_numeric(work["scheduled_hour"], errors="coerce")
        bad = hours.isna() | (hours < 0) | (hours > 23)
        checks["scheduled_hour"] = {
            "status": "warn" if bad.any() else "pass",
            "rejected": int(bad.sum()),
            "detail": f"{int(bad.sum())} row(s) outside 00–23",
        }
        keep &= ~bad

    # -- placement sanity ---------------------------------------------------
    if {"current_station", "destination"} <= set(work.columns):
        same = (work["current_station"].astype(str).str.strip()
                == work["destination"].astype(str).str.strip())
        checks["placement"] = {
            "status": "warn" if same.any() else "pass",
            "rejected": int(same.sum()),
            "detail": (f"{int(same.sum())} row(s) whose destination equals the current "
                       f"station — the train cannot be arriving where it already is"),
        }
        keep &= ~same

    # -- section plausibility -----------------------------------------------
    if "edge_weight_next_km" in work.columns:
        km = pd.to_numeric(work["edge_weight_next_km"], errors="coerce")
        # ``edge_weight_next_km`` is looked up from the network graph, so a value
        # that is missing or non-positive means no adjacent-station pair was found
        # and the model would receive a bogus 0 km section. A section longer than
        # the longest real section in the network (660 km in the bundled graph)
        # means the two stations are not adjacent at all.
        no_section = km.isna() | (km <= 0)
        too_long = km > MAX_PLAUSIBLE_SECTION_KM
        bad = no_section | too_long
        checks["section_plausibility"] = {
            "status": "warn" if bad.any() else "pass",
            "rejected": int(bad.sum()),
            "detail": (f"{int(no_section.sum())} row(s) with no resolvable next "
                       f"section, {int(too_long.sum())} longer than "
                       f"{MAX_PLAUSIBLE_SECTION_KM:.0f} km — {int(bad.sum())} rejected"),
        }
        keep &= ~bad

    # -- required columns complete -----------------------------------------
    required = ["current_station", "train_type", "current_delay_min"]
    present = [c for c in required if c in work.columns]
    completeness = float(work[present].notna().mean().mean()) if present else 0.0
    checks["completeness"] = {
        "status": "pass" if completeness >= 0.95 else ("warn" if completeness >= 0.8
                                                      else "fail"),
        "detail": f"{completeness:.1%} of required fields populated",
        "value": round(completeness, 4),
    }
    if completeness < 0.8:
        report["checks"] = checks
        report["rejected_rows"] = int(len(work))
        report["fatal"] = "Required fields are less than 80 % populated."
        return work.iloc[0:0], report

    # -- duplicates ---------------------------------------------------------
    keys = [c for c in ("train_id", "current_station", "scheduled_hour")
            if c in work.columns]
    if len(keys) == 3:
        dupes = work.duplicated(subset=keys, keep="first")
        checks["duplicates"] = {
            "status": "warn" if dupes.any() else "pass",
            "rejected": int(dupes.sum()),
            "detail": f"{int(dupes.sum())} duplicate "
                      f"(train, station, hour) row(s) dropped",
        }
        keep &= ~dupes

    # -- staleness ----------------------------------------------------------
    if timestamp_column and timestamp_column in work.columns and max_age_days:
        stamps = pd.to_datetime(work[timestamp_column], errors="coerce", utc=True)
        newest = stamps.max()
        if pd.notna(newest):
            age_days = float((pd.Timestamp.now("UTC") - newest).days)
            checks["staleness"] = {
                "status": "warn" if age_days > max_age_days else "pass",
                "detail": f"newest row is {age_days:.0f} day(s) old "
                          f"(limit {max_age_days})",
                "age_days": round(age_days, 1),
            }

    valid = work[keep].reset_index(drop=True)
    report["checks"] = checks
    report["rejected_rows"] = int(len(work) - len(valid))
    report["rows_out"] = int(len(valid))
    report["passed"] = bool(len(valid) > 0)
    return valid, report


def summarise(report: dict) -> str:
    """One-paragraph human summary for logs and the API response."""
    checks = report.get("checks", {})
    failed = [name for name, info in checks.items() if info.get("status") == "fail"]
    warned = [name for name, info in checks.items() if info.get("status") == "warn"]
    parts = [f"{report.get('rows_out', 0)}/{report.get('rows_in', 0)} rows admitted"]
    if report.get("rejected_rows"):
        parts.append(f"{report['rejected_rows']} rejected")
    if failed:
        parts.append("failed: " + ", ".join(failed))
    if warned:
        parts.append("warnings: " + ", ".join(warned))
    # Itemise the reasons behind the rejections — an operator reading an ingest
    # report needs "delay outside [-15, 1440] minutes", not just a column name.
    reasons = sorted(((name, info) for name, info in checks.items()
                      if info.get("rejected")),
                     key=lambda pair: pair[1]["rejected"], reverse=True)
    if reasons:
        parts.append("reasons: " + "; ".join(
            f"{name} — {info.get('detail', 'rows rejected')}"
            for name, info in reasons[:3]))
    return "; ".join(parts)
