"""
Scheduled timetable reconstructed from the journey record.

Why this exists
---------------
The dataset stores **one row per train** — a snapshot: where the train is, at what
scheduled hour, and where it is going. That is enough to forecast *one* train's
delay, but it cannot answer the dispatcher's next question: *"and which other
trains will this hold up?"* Answering that needs a schedule: who is on which
section, and when.

So the timetable is reconstructed explicitly and stored, rather than guessed at
query time:

* the path is the network's shortest route stitched through
  ``origin → current_station → destination`` (the same routing the app already
  uses for ``/gps/route``);
* the train's recorded ``scheduled_hour`` anchors it at ``current_station``;
* run time between stations comes from a per-type commercial speed plus a dwell
  at each stop.

**Everything the timetable adds beyond the dataset is an assumption, and every
assumption is named here and reported by the API.** Commercial speeds and dwell
times are order-of-magnitude figures for Indian Railways stock; they set the
*resolution* of a conflict, not its existence. The dataset is itself simulated,
so the right claim for this module is "a self-consistent schedule, derived from
the recorded journeys with stated running-time assumptions" — not "the real
SWR timetable", which is not publicly available.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional

from config import PROCESSED_DIR, RAW_CSV
from graph_utils import build_graph, path_distance_km, shortest_path_km

TIMETABLE_JSON = PROCESSED_DIR / "timetable.json"

#: Commercial (start-to-stop, including stops) speeds in km/h by service type.
#: ASSUMED values — used to place a train between the stations it is recorded at.
COMMERCIAL_SPEED_KMH: dict[str, float] = {
    "Superfast": 72.0,
    "Express": 62.0,
    "Intercity": 56.0,
    "Passenger": 44.0,
    "MEMU": 40.0,
    "Freight": 32.0,
}
DEFAULT_SPEED_KMH = 55.0

#: Minutes spent at an intermediate stop, by type. ASSUMED.
DWELL_MIN: dict[str, float] = {
    "Superfast": 2.0, "Express": 2.0, "Intercity": 1.5,
    "Passenger": 1.0, "MEMU": 0.75, "Freight": 4.0,
}
DEFAULT_DWELL_MIN = 2.0

ASSUMPTIONS = {
    "source": "reconstructed from data/raw/swr_journeys.csv (one snapshot per train)",
    "routing": "network shortest path stitched through origin → current_station → destination",
    "anchor": "the recorded scheduled_hour is the time at current_station",
    "commercial_speed_kmh": COMMERCIAL_SPEED_KMH,
    "dwell_min_per_stop": DWELL_MIN,
    "caveat": ("commercial speeds and dwell times are assumed; the timetable is "
               "self-consistent but is not the real SWR working timetable"),
}


def speed_for(train_type: str) -> float:
    return COMMERCIAL_SPEED_KMH.get(train_type, DEFAULT_SPEED_KMH)


def dwell_for(train_type: str) -> float:
    return DWELL_MIN.get(train_type, DEFAULT_DWELL_MIN)


def run_time_min(km: float, train_type: str) -> float:
    """Minutes to cover ``km`` at the type's commercial speed."""
    return 60.0 * max(km, 0.0) / max(speed_for(train_type), 1.0)


@dataclass(frozen=True)
class ScheduledStop:
    station: str
    minute: float          # minutes after midnight, can exceed 1440 for next-day


@dataclass(frozen=True)
class TimetableEntry:
    """One train, placed at every station on its path."""
    train_id: str
    train_type: str
    day_of_week: int
    origin: str
    destination: str
    anchor_station: str
    anchor_minute: float
    stops: tuple[ScheduledStop, ...]
    path_km: tuple[float, ...]      # cumulative km from the path's first station

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(stop.station for stop in self.stops)

    def index_of(self, station: str) -> Optional[int]:
        for index, stop in enumerate(self.stops):
            if stop.station == station:
                return index
        return None

    def minute_at(self, station: str) -> Optional[float]:
        index = self.index_of(station)
        return None if index is None else self.stops[index].minute

    def sections(self) -> Iterable[tuple[str, str, float, float]]:
        """``(from, to, entry_minute, exit_minute)`` for each section traversed."""
        for index in range(len(self.stops) - 1):
            start = self.stops[index]
            end = self.stops[index + 1]
            yield start.station, end.station, start.minute, end.minute


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def _stitch_path(graph, origin: str, current: str, destination: str) -> list[str]:
    """origin → current → destination as one station list, without duplicates."""
    path: list[str] = []
    if origin in graph and current in graph and origin != current:
        head = shortest_path_km(graph, origin, current) or [current]
        path.extend(head[:-1])
    path.append(current)
    if destination not in graph or destination == current:
        return path
    tail = shortest_path_km(graph, current, destination) or []
    path.extend(station for station in tail if station != current)
    return path


def _schedule_for(graph, row) -> Optional[dict]:
    """Place one recorded train at every station on its stitched path."""
    train_type = str(row.get("train_type", "Express"))
    origin = str(row.get("origin"))
    current = str(row.get("current_station"))
    destination = str(row.get("destination"))
    if current not in graph:
        return None

    path = _stitch_path(graph, origin, current, destination)
    if len(path) < 2:
        return None

    anchor_index = path.index(current)
    anchor_minute = float(row.get("scheduled_hour", 0)) * 60.0

    # Walk backwards to the origin, then forwards to the destination, so the whole
    # path has times consistent with the anchor.
    minutes = [0.0] * len(path)
    minutes[anchor_index] = anchor_minute
    for index in range(anchor_index - 1, -1, -1):
        km = float(graph.edges[path[index], path[index + 1]]["km"])
        minutes[index] = minutes[index + 1] - run_time_min(km, train_type) - dwell_for(train_type)
    for index in range(anchor_index + 1, len(path)):
        km = float(graph.edges[path[index - 1], path[index]]["km"])
        minutes[index] = (minutes[index - 1] + dwell_for(train_type)
                          + run_time_min(km, train_type))

    cumulative = [0.0]
    for index in range(len(path) - 1):
        cumulative.append(cumulative[-1]
                          + float(graph.edges[path[index], path[index + 1]]["km"]))

    return {
        "train_id": str(row.get("train_id")),
        "train_type": train_type,
        "day_of_week": int(row.get("day_of_week", 0)),
        "origin": origin, "destination": destination,
        "anchor_station": current, "anchor_minute": anchor_minute,
        "stations": path,
        "minutes": [round(value, 1) for value in minutes],
        "km": [round(value, 2) for value in cumulative],
    }


def build_timetable(*, force: bool = False, limit: Optional[int] = None) -> dict:
    """Build the timetable from the journey record and cache it as JSON."""
    if TIMETABLE_JSON.exists() and not force:
        return json.loads(TIMETABLE_JSON.read_text())

    import pandas as pd

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} is missing — run src/data_generator.py first.")
    frame = pd.read_csv(RAW_CSV)
    if limit:
        frame = frame.head(limit)

    graph = build_graph()
    trains = []
    for record in frame.to_dict("records"):
        scheduled = _schedule_for(graph, record)
        if scheduled is not None:
            trains.append(scheduled)

    payload = {"assumptions": ASSUMPTIONS, "count": len(trains), "trains": trains}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMETABLE_JSON.write_text(json.dumps(payload))
    return payload


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------
class Timetable:
    """In-memory view of the timetable, indexed by station for fast windows.

    ``day`` is the dataset's day index (0–6) *or* ``None`` to ignore the day,
    which is what a dispatcher wants for a "today"-style question when the day is
    not known.
    """

    def __init__(self, payload: dict):
        self.assumptions = payload.get("assumptions", ASSUMPTIONS)
        self.entries: list[TimetableEntry] = []
        self.by_id: dict[str, TimetableEntry] = {}
        self._by_station: dict[str, list[tuple[TimetableEntry, int]]] = {}

        for record in payload.get("trains", []):
            stops = tuple(ScheduledStop(station, minute) for station, minute
                          in zip(record["stations"], record["minutes"]))
            entry = TimetableEntry(
                train_id=record["train_id"], train_type=record["train_type"],
                day_of_week=int(record["day_of_week"]),
                origin=record["origin"], destination=record["destination"],
                anchor_station=record["anchor_station"],
                anchor_minute=float(record["anchor_minute"]),
                stops=stops, path_km=tuple(record["km"]),
            )
            self.entries.append(entry)
            self.by_id.setdefault(entry.train_id, entry)
            for index, stop in enumerate(stops):
                self._by_station.setdefault(stop.station, []).append((entry, index))

    def __len__(self) -> int:
        return len(self.entries)

    def stations(self) -> list[str]:
        return sorted(self._by_station)

    def at_station(self, station: str, start: float, end: float,
                   day: Optional[int] = None) -> list[tuple[TimetableEntry, float]]:
        """Every train present at ``station`` within ``[start, end)``.

        Times are minutes after midnight and may exceed 1440 for the small hours
        of the next day, so a window that wraps is handled by testing both.
        """
        found: list[tuple[TimetableEntry, float]] = []
        for entry, index in self._by_station.get(station, ()):
            if day is not None and entry.day_of_week != day:
                continue
            minute = entry.stops[index].minute
            for candidate in (minute, minute + 1440.0, minute - 1440.0):
                if start <= candidate < end:
                    found.append((entry, candidate))
                    break
        found.sort(key=lambda pair: pair[1])
        return found

    def section_traffic(self, from_station: str, to_station: str,
                        day: Optional[int] = None
                        ) -> list[tuple[TimetableEntry, float, float]]:
        """``(train, entry_minute, exit_minute)`` for one directed section."""
        found = []
        for entry in self.entries:
            if day is not None and entry.day_of_week != day:
                continue
            for source, target, entry_minute, exit_minute in entry.sections():
                if source == from_station and target == to_station:
                    found.append((entry, entry_minute, exit_minute))
                    break
        found.sort(key=lambda item: item[1])
        return found

    def departing_from(self, station: str, day: Optional[int] = None,
                       within_minutes: Optional[float] = None,
                       after: Optional[float] = None) -> list[tuple[TimetableEntry, float]]:
        """Trains whose *journey begins* at ``station`` (candidate connections)."""
        found = []
        for entry in self.entries:
            if day is not None and entry.day_of_week != day:
                continue
            if entry.stops[0].station != station:
                continue
            depart = entry.stops[0].minute
            if after is not None and depart < after:
                continue
            if within_minutes is not None and after is not None \
                    and depart > after + within_minutes:
                continue
            found.append((entry, depart))
        found.sort(key=lambda item: item[1])
        return found


_CACHE: Optional[Timetable] = None


def load_timetable(*, rebuild: bool = False) -> Timetable:
    """Load the cached timetable (building it on first use)."""
    global _CACHE
    if _CACHE is None or rebuild:
        _CACHE = Timetable(build_timetable(force=rebuild))
    return _CACHE
