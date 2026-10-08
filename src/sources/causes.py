"""
Delay-cause taxonomy.

Indian Railways attributes every lost minute to a cause head in its punctuality
statements. A predictor that only says *"you will arrive 48 minutes late"* cannot
help anyone prevent it; the cause is what tells the control office whether to
escalate to the power (locomotive) controller, the engineering department, or
operations. Ingesting causes turns the app from a delay reporter into an
attribution tool: *"61 % of the delay-minutes on this section this week were
rolling-stock failures"* is an actionable statement, and it is the kind of
evidence a real improvement plan needs.

Mapping is deliberately loose: operators use differing abbreviations, so the
normaliser accepts the common variants and falls back to ``UNCLASSIFIED`` rather
than guessing.
"""
from __future__ import annotations

# Canonical heads (mirroring the lettered cause heads used in IR punctuality
# statements) + how each is escalated operationally.
CAUSE_HEADS: dict[str, dict] = {
    "PRE_OCCUPIED_LINE": {
        "label": "Pre-occupied line / congestion",
        "owner": "Section Controller (operations)",
        "typical_fix": "Precedence change, loop-line crossing, headway smoothing",
    },
    "ROLLING_STOCK": {
        "label": "Rolling-stock / locomotive failure",
        "owner": "Power controller (shed)",
        "typical_fix": "Rake swap, banker attachment, shed maintenance escalation",
    },
    "ENGINEERING": {
        "label": "Engineering block / speed restriction (TRT, TLC)",
        "owner": "Engineering department",
        "typical_fix": "Block rescheduling, temporary-speed-restriction review",
    },
    "CREW": {
        "label": "Crew / duty-hour (HOER) constraint",
        "owner": "Crew controller (running room)",
        "typical_fix": "Crew relief at next junction, spare-crew induction",
    },
    "SIGNALLING": {
        "label": "Signal / interlocking failure",
        "owner": "Signal & Telecom",
        "typical_fix": "Panel repair escalation, single-line working, caution order",
    },
    "NATURAL": {
        "label": "Weather / natural cause (fog, flood, landslide)",
        "owner": "Operations + safety",
        "typical_fix": "Fog patching, speed restriction, diversion",
    },
    "CONVENTION": {
        "label": "Convention / connection / administrative",
        "owner": "Section Controller",
        "typical_fix": "Connection-hold policy review, timetable adjustment",
    },
    "OTHER": {
        "label": "Other / miscellaneous",
        "owner": "Section Controller",
        "typical_fix": "Case-by-case review",
    },
    "UNCLASSIFIED": {
        "label": "Unclassified (missing or unrecognised cause)",
        "owner": "Data governance",
        "typical_fix": "Improve cause recording at source",
    },
}

# Raw token -> canonical head. Matching is substring-based on the lower-cased,
# punctuation-stripped raw value, so "RS FAILURE", "loco failure" and
# "B/RS/FAILURE" all land on ROLLING_STOCK.
_ALIASES: dict[str, str] = {
    # pre-occupied line / congestion
    "a/": "PRE_OCCUPIED_LINE", "preoccupied": "PRE_OCCUPIED_LINE",
    "pre-occupied": "PRE_OCCUPIED_LINE", "pre occupied": "PRE_OCCUPIED_LINE",
    "congestion": "PRE_OCCUPIED_LINE", "line clearance": "PRE_OCCUPIED_LINE",
    "late start": "PRE_OCCUPIED_LINE", "detention": "PRE_OCCUPIED_LINE",
    "path": "PRE_OCCUPIED_LINE", "crossing": "PRE_OCCUPIED_LINE",
    # rolling stock
    "b/": "ROLLING_STOCK", "loco": "ROLLING_STOCK", "rs failure": "ROLLING_STOCK",
    "rolling stock": "ROLLING_STOCK", "rolling-stock": "ROLLING_STOCK",
    "engine": "ROLLING_STOCK", "wagon": "ROLLING_STOCK", "rake": "ROLLING_STOCK",
    "brake": "ROLLING_STOCK", "sick": "ROLLING_STOCK", "failure": "ROLLING_STOCK",
    # engineering
    "c/": "ENGINEERING", "engg": "ENGINEERING", "engineering": "ENGINEERING",
    "trt": "ENGINEERING", "tlc": "ENGINEERING", "tsr": "ENGINEERING",
    "block": "ENGINEERING", "permanent way": "ENGINEERING", "caution": "ENGINEERING",
    # crew
    "crew": "CREW", "hoer": "CREW", "duty": "CREW", "loco pilot": "CREW",
    "running staff": "CREW", "signing on": "CREW",
    # signalling
    "signal": "SIGNALLING", "snt": "SIGNALLING", "interlock": "SIGNALLING",
    "point failure": "SIGNALLING", "panel": "SIGNALLING", "block instrument": "SIGNALLING",
    # natural
    "fog": "NATURAL", "rain": "NATURAL", "flood": "NATURAL", "storm": "NATURAL",
    "weather": "NATURAL", "landslide": "NATURAL", "boulder": "NATURAL",
    "d/": "NATURAL",
    # convention / administrative
    "convention": "CONVENTION", "connection": "CONVENTION", "late supply": "CONVENTION",
    "e/": "CONVENTION", "commercial": "CONVENTION", "medical": "CONVENTION",
    "law and order": "CONVENTION", "agitation": "CONVENTION",
    # other
    "other": "OTHER", "misc": "OTHER", "f/": "OTHER", "g/": "OTHER",
}


def classify_cause(raw: object) -> str:
    """Map a raw operator string to a canonical cause head.

    Returns ``UNCLASSIFIED`` for missing/unknown input — never guesses, because a
    wrong cause attribution is worse than an absent one.
    """
    if raw is None:
        return "UNCLASSIFIED"
    text = str(raw).strip().lower()
    if not text or text in {"nan", "none", "null", "-"}:
        return "UNCLASSIFIED"

    # Exact canonical key?
    upper = text.upper().replace(" ", "_")
    if upper in CAUSE_HEADS:
        return upper

    # Longest alias match wins, so "loco pilot" beats "loco".
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if alias in text:
            return _ALIASES[alias]
    return "UNCLASSIFIED"


def describe(cause_head: str) -> dict:
    return CAUSE_HEADS.get(cause_head, CAUSE_HEADS["UNCLASSIFIED"])


def attribution(counts_by_cause: dict[str, int], minutes_by_cause: dict[str, float]) -> list[dict]:
    """Rank causes by delay-minutes, with the operational escalation for each."""
    total_minutes = sum(minutes_by_cause.values()) or 1.0
    rows = []
    for head, minutes in sorted(minutes_by_cause.items(),
                                key=lambda kv: kv[1], reverse=True):
        info = describe(head)
        rows.append({
            "cause": head,
            "label": info["label"],
            "events": int(counts_by_cause.get(head, 0)),
            "delay_minutes": round(float(minutes), 1),
            "share_pct": round(100.0 * float(minutes) / total_minutes, 1),
            "owner": info["owner"],
            "typical_fix": info["typical_fix"],
        })
    return rows
