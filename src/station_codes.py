"""
Station-code resolution (shared by the API and the data-ingestion layer).

Real railway data arrives keyed by **station code** (``SBC``, ``MYS``, ``UBL``)
while the graph is keyed by station *name*. The original code kept this mapping
buried inside ``app.py``; it lives here so that ingestion, validation and the API
all resolve codes the same way — a mismatch between them was previously a silent
source of dropped rows.
"""
from __future__ import annotations

import re

# Real Indian Railways codes for the stations in the graph.
REAL_CODES: dict[str, str] = {
    "KSR Bengaluru": "SBC", "Bengaluru Cantonment": "BNC", "Yesvantpur": "YPR",
    "Krishnarajapuram": "KJM", "Whitefield": "WFD", "Yelahanka": "YNK",
    "Kengeri": "KGI", "Bidadi": "BID", "Ramanagara": "RMGM", "Channapatna": "CPT",
    "Maddur": "MAD", "Mandya": "MYA", "Srirangapatna": "S", "Mysuru": "MYS",
    "Nanjangud": "NTW", "Chamarajanagar": "CMNR", "Krishnarajanagara": "KRNR",
    "Hassan": "HAS", "Sakleshpur": "SKLR", "Arsikere": "ASK", "Tumakuru": "TK",
    "Tiptur": "TTR", "Birur": "RRB", "Davangere": "DVG", "Harihar": "HRR",
    "Haveri": "HVR", "Hubballi": "UBL", "Dharwad": "DWR", "Belagavi": "BGM",
    "Gadag": "GDG", "Hosapete": "HPT", "Ballari": "BAY", "Hosur": "HSRA",
    "Kolar": "KQZ", "Bangarapet": "BWT", "Malur": "MLO", "Chitradurga": "CTA",
    "Chennai Central": "MAS", "Renigunta": "RU", "Tirupati": "TPTY",
    "Vijayawada": "BZA", "Visakhapatnam": "VSKP", "Secunderabad": "SC",
    "Warangal": "WL", "Guntakal": "GTL", "Raichur": "RC", "Kalaburagi": "KLBG",
    "Solapur": "SUR", "Nagpur": "NGP", "Bhopal": "BPL", "Agra Cantt": "AGC",
    "Hazrat Nizamuddin": "NZM", "Jaipur": "JP", "Ahmedabad": "ADI",
    "Mumbai CSMT": "CSMT", "Pune": "PUNE", "Coimbatore": "CBE", "Erode": "ED",
    "Salem": "SA", "Tiruchirappalli": "TPJ", "Madurai": "MDU",
    "Thiruvananthapuram": "TVC", "Ernakulam": "ERS", "Kozhikode": "CLT",
    "Mangaluru Central": "MAQ", "Palakkad": "PGT", "Aurangabad": "AWB",
}

CODE_TO_NAME: dict[str, str] = {v: k for k, v in REAL_CODES.items()}

_STOPWORDS = {"of", "the", "and", "jr", "jn", "road", "cantt", "central",
              "junction", "city", "town"}


def station_code(name: str, used: set[str] | None = None) -> str:
    """Return a unique station code — the real IR code where known, else derived."""
    used = used if used is not None else set()
    if name in REAL_CODES:
        code, base, n = REAL_CODES[name], REAL_CODES[name], 1
        while code in used:
            n += 1
            code = f"{base}{n}"
        used.add(code)
        return code

    words = [w for w in re.split(r"[^A-Za-z0-9]+", name) if w]
    if not words:
        return name[:4].upper()
    initials = [w[0].upper() for w in words if w.lower() not in _STOPWORDS]
    code = "".join(initials)[:4] if len(initials) >= 2 else words[0][:4].upper()
    base, n = code, 1
    while code in used:
        n += 1
        code = f"{base[:3]}{n}"
    used.add(code)
    return code


def all_codes() -> dict[str, str]:
    """Name → code for every known station."""
    used: set[str] = set()
    return {name: station_code(name, used) for name in REAL_CODES}


def resolve_station(token: str, station_names: list[str]) -> str | None:
    """Resolve a station name *or* code (any case) to a graph node name.

    Handles the three shapes real exports use: the exact name, the official
    code, and a loose case/space-insensitive variant.
    """
    if not token:
        return None
    raw = str(token).strip()
    if not raw:
        return None

    by_name = {s.lower(): s for s in station_names}
    if raw.lower() in by_name:
        return by_name[raw.lower()]

    upper = raw.upper()
    if upper in CODE_TO_NAME and CODE_TO_NAME[upper] in station_names:
        return CODE_TO_NAME[upper]

    squashed = re.sub(r"[^a-z0-9]", "", raw.lower())
    for name in station_names:
        if re.sub(r"[^a-z0-9]", "", name.lower()) == squashed:
            return name

    # last resort: derived code (handles stations without an official code)
    used: set[str] = set()
    for name in station_names:
        if station_code(name, used).upper() == upper:
            return name
    return None
