"""
FastAPI backend for the Railway Dispatcher Dashboard.

Loads the static NetworkX graph + trained bundle into memory, computes dynamic
centralities on ping, serves prediction endpoints, and handles CSV exports.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000   (from the project root)
"""
from __future__ import annotations

import io
import math
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests as _tile_requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AUTH_PASS, AUTH_USER, DATA_DIR, EXPORTS_DIR, RESULTS_DIR, ROOT
from graph_utils import path_distance_km, shortest_path_km
from inference import CascadePredictor

app = FastAPI(
    title="SWR Cascade Delay API",
    description="Graph-Derived Feature Boosting for Cascading Train Delay Prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Static assets (figures, result tables, sample CSV) for the dashboard.
app.mount("/static", StaticFiles(directory=str(RESULTS_DIR)), name="static")
app.mount("/data-static", StaticFiles(directory=str(DATA_DIR)), name="data-static")

# Leaflet.js + map assets (served locally so the map works without a CDN).
_LIB_DIR = ROOT / "lib"
_LIB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/lib", StaticFiles(directory=str(_LIB_DIR)), name="lib")


# ---------------------------------------------------------------------------
# Same-origin tile proxy.
#
# OpenStreetMap's volunteer tile servers return a literal "403 Access blocked"
# tile image to clients that don't follow their usage policy — this happens
# routinely when the *browser* requests tiles while carrying the Referer of a
# hosting/proxy domain (the user saw exactly those blocked tiles). The fix:
# the browser asks THIS app for tiles (same origin, no external referer) and
# the app performs the real upstream tile request with a proper identifying
# User-Agent, caching the result. Every zoom level still issues real tile
# requests; they are simply proxied so providers never see a blockable client.
# ---------------------------------------------------------------------------
_TILE_UA = "RailPulse/1.0 (educational train-delay demo, tile proxy)"
_TILE_CACHE: dict = {}
_TILE_CACHE_MAX = 800


def _tile_fetch(urls: list[str], z: int, x: int, y: int):
    for url in urls:
        u = url.format(z=z, x=x, y=y)
        try:
            r = _tile_requests.get(u, timeout=8, headers={"User-Agent": _TILE_UA})
            ctype = (r.headers.get("content-type") or "").split(";")[0]
            if r.status_code == 200 and r.content and ctype.startswith("image/"):
                return r.content, ctype
        except Exception:
            continue
    return None, None


def _serve_tile(key: str, urls: list[str], z: int, x: int, y: int):
    hit = _TILE_CACHE.get(key)
    if hit is None:
        body, ctype = _tile_fetch(urls, z, x, y)
        if body is None:
            raise HTTPException(404, "tile unavailable")
        if len(_TILE_CACHE) >= _TILE_CACHE_MAX:
            _TILE_CACHE.clear()
        _TILE_CACHE[key] = (body, ctype)
    else:
        body, ctype = hit
    return Response(
        content=body, media_type=ctype,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/tiles/osm/{z}/{x}/{y}.png")
def tile_osm(z: int, x: int, y: int):
    """Standard OSM raster tiles, proxied; falls back to Esri light canvas."""
    return _serve_tile(
        f"osm/{z}/{x}/{y}",
        [
            "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        ], z, x, y)


@app.get("/tiles/esri/dark-base/{z}/{y}/{x}.png")
def tile_esri_dark_base(z: int, y: int, x: int):
    """Esri Dark Gray canvas (Google-Maps-dark look), proxied."""
    return _serve_tile(
        f"esd/{z}/{y}/{x}",
        [
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        ], z, x, y)


@app.get("/tiles/esri/dark-ref/{z}/{y}/{x}.png")
def tile_esri_dark_ref(z: int, y: int, x: int):
    """Esri Dark Gray reference (place labels) overlay, proxied."""
    return _serve_tile(
        f"esr/{z}/{y}/{x}",
        [
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        ], z, x, y)


def _station_code(name: str, used: set[str]) -> str:
    """Return a unique station code — real IR code where known, else derived."""
    # Real Indian Railways station codes for well-known stations.
    REAL_CODES = {
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
    if name in REAL_CODES:
        code = REAL_CODES[name]
        base = code
        n = 1
        while code in used:
            n += 1
            code = f"{base}{n}"
        used.add(code)
        return code
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name) if w]
    if not words:
        return name[:4].upper()
    # Prefer initials of meaningful words (skip tiny stopwords).
    stop = {"of", "the", "and", "jr", "jn", "road", "cantt", "central", "junction", "city", "town"}
    initials = [w[0].upper() for w in words if w.lower() not in stop]
    if len(initials) >= 2:
        code = "".join(initials)[:4]
    else:
        code = words[0][:4].upper()
    base = code
    n = 1
    while code in used:
        n += 1
        code = f"{base[:3]}{n}"
    used.add(code)
    return code

predictor = CascadePredictor()

# Dispatcher Copilot — natural-language agent over the live predictor/graph.
from copilot import DispatcherCopilot  # noqa: E402

copilot = DispatcherCopilot(predictor)

# ---------------------------------------------------------------------------
# In-memory user store + prediction history (demo-grade; resets on restart).
# ---------------------------------------------------------------------------
USERS = {
    AUTH_USER: {"password": AUTH_PASS, "full_name": "Dispatcher", "email": "admin@swr.in"},
}
HISTORY: list[dict] = []


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points, in kilometres."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_station(lat: float, lon: float) -> tuple[str, float]:
    """Return (station_name, distance_km) of the graph node closest to (lat, lon)."""
    best, best_d = None, float("inf")
    for n in predictor.G.nodes():
        slat = predictor.G.nodes[n]["lat"]
        slon = predictor.G.nodes[n]["lon"]
        d = _haversine_km(lat, lon, slat, slon)
        if d < best_d:
            best, best_d = n, d
    return best, best_d


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ManualRequest(BaseModel):
    current_station: str
    upcoming_station: Optional[str] = None
    destination: Optional[str] = None
    train_type: str = "Express"
    train_number: Optional[str] = None
    hour: int = 14
    day: str = "Monday"
    weather: str = "Clear"
    current_delay_min: float = 0.0
    model: Optional[str] = None



class RegisterRequest(BaseModel):
    full_name: str
    email: str
    username: str
    password: str


class BatchRow(BaseModel):
    current_station: str
    train_type: str = "Express"
    hour: int = 14
    day: str = "Monday"
    weather: str = "Clear"
    current_delay_min: float = 0.0
    destination: Optional[str] = None
    upcoming_station: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class CopilotRequest(BaseModel):
    message: str


class GpsRequest(BaseModel):
    lat: float
    lon: float
    speed_kmh: float = 60.0
    train_number: Optional[str] = None
    destination: Optional[str] = None
    train_type: str = "Express"
    day: str = "Monday"
    weather: str = "Clear"
    hour: Optional[int] = None
    current_delay_min: float = 0.0
    model: Optional[str] = None


class SelectModelRequest(BaseModel):
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Serve the standalone dispatcher dashboard (proxy/iframe-friendly)."""
    return FileResponse(str(Path(__file__).parent / "dashboard.html"))


# ---------------------------------------------------------------------------
# Dispatcher Copilot (agentic Q&A over the live network)
# ---------------------------------------------------------------------------
@app.get("/copilot/status")
def copilot_status():
    return copilot.status()


@app.post("/copilot/chat")
def copilot_chat(req: CopilotRequest):
    return copilot.chat(req.message)


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.dashboard_name}


@app.post("/login")
def login(creds: LoginRequest):
    """Validate dispatcher credentials (in-memory user store)."""
    u = creds.username.strip().lower()
    rec = next((v for k, v in USERS.items() if k.lower() == u), None)
    if rec and rec["password"] == creds.password:
        return {"success": True, "username": creds.username, "full_name": rec["full_name"]}
    return {"success": False}


@app.post("/register")
def register(req: RegisterRequest):
    """Create a new dispatcher account (in-memory).

    Enforces password + email restrictions:
      - password: 8+ chars, at least one uppercase, one lowercase, one digit,
        one special character
      - email: must be a valid address (user@domain.tld)
    """
    u = req.username.strip()
    if not u or not req.password:
        return {"success": False, "message": "Username and password are required."}
    if len(u) < 3:
        return {"success": False, "message": "Username must be at least 3 characters."}
    email = (req.email or "").strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return {"success": False, "message":
                "Enter a valid email address (e.g. name@railways.in)."}
    p = req.password
    issues = []
    if len(p) < 8:
        issues.append("8+ characters")
    if not re.search(r"[A-Z]", p):
        issues.append("one uppercase letter")
    if not re.search(r"[a-z]", p):
        issues.append("one lowercase letter")
    if not re.search(r"\d", p):
        issues.append("one digit")
    if not re.search(r"[^A-Za-z0-9]", p):
        issues.append("one special character")
    if issues:
        return {"success": False, "message":
                "Password needs " + ", ".join(issues) + "."}
    if any(k.lower() == u.lower() for k in USERS):
        return {"success": False, "message": "Username already exists."}
    USERS[u] = {"password": req.password, "full_name": req.full_name.strip() or u,
                "email": email}
    return {"success": True, "username": u, "full_name": USERS[u]["full_name"]}


@app.get("/system/overview")
def system_overview():
    metrics = predictor.model_metrics()
    best = predictor.best_name
    return {
        "model": predictor.dashboard_name,
        "best_single_model": best,
        "best_mae": round(float(predictor.bundle.get("best_mae", 0.0)), 3),
        "n_stations": predictor.G.number_of_nodes(),
        "n_edges": predictor.G.number_of_edges(),
        "n_features": len(predictor.features),
        "target": predictor.bundle.get("metadata", {}).get("target", "?"),
        "n_train": predictor.bundle.get("metadata", {}).get("n_train", 0),
        "n_test": predictor.bundle.get("metadata", {}).get("n_test", 0),
        "network_version": predictor.bundle.get("metadata", {}).get("network_version", "?"),
        "available_models": [m["name"] for m in predictor.available_models()],
    }


@app.get("/graph")
def graph():
    """The full rail network as JSON: nodes (with all centralities) + edges."""
    G = predictor.G
    cent = predictor.cent
    used_codes: set[str] = set()
    nodes = []
    for n in G.nodes():
        c = cent.get(n, {})
        nodes.append({
            "id": n,
            "code": _station_code(n, used_codes),
            "lat": float(G.nodes[n]["lat"]),
            "lon": float(G.nodes[n]["lon"]),
            "degree": int(G.degree(n)),
            "degree_centrality": round(float(c.get("degree_centrality", 0.0)), 6),
            "eigenvector_centrality": float(c.get("eigenvector_centrality", 0.0)),
            "pagerank": round(float(c.get("pagerank", 0.0)), 6),
            "betweenness_centrality": round(float(c.get("betweenness_centrality", 0.0)), 6),
            "closeness_centrality": round(float(c.get("closeness_centrality", 0.0)), 6),
        })
    edges = [{"source": a, "target": b, "km": float(G[a][b]["km"])} for a, b in G.edges()]
    return {"nodes": nodes, "edges": edges, "layout": "geo"}


@app.get("/stations")
def stations():
    return {"stations": predictor.station_names()}


@app.get("/stations/{name}")
def station_meta(name: str):
    if name not in predictor.G:
        raise HTTPException(404, f"Unknown station '{name}'")
    return predictor.station_meta(name)


@app.get("/model/metrics")
def model_metrics():
    return predictor.model_metrics()


@app.get("/models")
def models():
    """The selectable model menu (top-5 persisted models) + active model."""
    return {"models": predictor.available_models(),
            "active": predictor.dashboard_name}


@app.post("/model/select")
def model_select(req: SelectModelRequest):
    """Switch the active prediction model used by the whole app."""
    try:
        name = predictor.set_model(req.model)
    except KeyError:
        raise HTTPException(
            404, f"Unknown model '{req.model}'. "
                 f"Available: {', '.join(sorted(predictor.models))}")
    return {"success": True, "active": name,
            "models": predictor.available_models()}


@app.post("/predict/manual")
def predict_manual(req: ManualRequest):
    if req.current_station not in predictor.G:
        raise HTTPException(404, f"Unknown station '{req.current_station}'")
    if req.model:  # optional per-request model override (live menu switch)
        try:
            predictor.set_model(req.model)
        except KeyError:
            raise HTTPException(404, f"Unknown model '{req.model}'")
    result = predictor.predict(
        current_station=req.current_station,
        upcoming_station=req.upcoming_station,
        destination=req.destination,
        train_type=req.train_type, hour=req.hour, day=req.day,
        weather=req.weather, current_delay_min=req.current_delay_min,
    )
    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    HISTORY.append({
        "timestamp": result["timestamp"],
        "current_station": req.current_station,
        "destination": req.destination or req.current_station,
        "train_type": req.train_type,
        "train_number": req.train_number or "",
        "current_delay_min": req.current_delay_min,
        "predicted_delay_min": result["predicted_destination_arrival_delay_min"],
        "alert_level": result["alert"]["level"],
    })
    HISTORY[:] = HISTORY[-50:]
    return result


@app.get("/history")
def history():
    return {"history": list(reversed(HISTORY))}


# ---------------------------------------------------------------------------
# Live GPS train tracking (reuses the same NetworkX graph + CascadePredictor)
# Flow: GPS (lat,lon) -> nearest station -> current delay -> ML prediction
#       -> NetworkX cascade -> updated risk -> map
# ---------------------------------------------------------------------------
@app.get("/gps/nearest")
def gps_nearest(lat: float, lon: float):
    """Detect the nearest railway station for a GPS coordinate."""
    name, dist = _nearest_station(lat, lon)
    return {
        "station": name,
        "distance_km": round(dist, 2),
        "lat": predictor.G.nodes[name]["lat"],
        "lon": predictor.G.nodes[name]["lon"],
    }


@app.get("/gps/route")
def gps_route(origin: str, destination: str):
    """Shortest rail route (NetworkX) with coordinates — used by Demo GPS mode."""
    if origin not in predictor.G:
        raise HTTPException(404, f"Unknown station '{origin}'")
    if destination not in predictor.G:
        raise HTTPException(404, f"Unknown station '{destination}'")
    path = shortest_path_km(predictor.G, origin, destination)
    if not path:
        raise HTTPException(404, "No route found between those stations")
    coords = [[predictor.G.nodes[s]["lat"], predictor.G.nodes[s]["lon"]] for s in path]
    return {"route": path, "coords": coords,
            "km": round(path_distance_km(predictor.G, path), 1)}


@app.post("/gps/predict")
def gps_predict(req: GpsRequest):
    """Full GPS→prediction→cascade pipeline for one train position.

    Passes min_delay_for_alert=0.0 so a real 0-minute delay is reported and
    displayed as 0 (never bumped to a minimum).
    """
    if req.model:
        try:
            predictor.set_model(req.model)
        except KeyError:
            raise HTTPException(404, f"Unknown model '{req.model}'")
    nearest, dist = _nearest_station(req.lat, req.lon)
    dest = req.destination or None
    upcoming = None
    route = None
    route_km = 0.0
    if dest and dest != nearest:
        path = shortest_path_km(predictor.G, nearest, dest)
        if path and len(path) > 1:
            upcoming = path[1]
            route = path
            route_km = path_distance_km(predictor.G, path)
    if upcoming is None:
        upcoming = predictor._default_upcoming(nearest, dest)
    hour = req.hour if req.hour is not None else int(time.strftime("%H"))

    res = predictor.predict(
        current_station=nearest,
        upcoming_station=upcoming,
        destination=dest,
        train_type=req.train_type,
        hour=hour,
        day=req.day,
        weather=req.weather,
        current_delay_min=req.current_delay_min,
        min_delay_for_alert=0.0,
    )
    alert = res["alert"]
    return {
        "gps": {"lat": req.lat, "lon": req.lon, "speed_kmh": req.speed_kmh},
        "nearest_station": nearest,
        "distance_to_station_km": round(dist, 2),
        "upcoming_station": upcoming,
        "destination": dest,
        "route": route,
        "route_km": round(route_km, 1),
        "train_number": req.train_number,
        "hour": hour,
        "current_delay_min": req.current_delay_min,
        "predicted_destination_arrival_delay_min": res["predicted_destination_arrival_delay_min"],
        "alert": alert,
        "model": res["model"],
    }


@app.get("/demo/records")
def demo_records():
    """Five pre-canned scenarios run live through the predictor."""
    scenarios = [
        dict(current_station="Mysuru", destination="KSR Bengaluru", train_type="Superfast",
             train_number="20807", hour=17, day="Monday", weather="Clear", current_delay_min=45),
        dict(current_station="KSR Bengaluru", destination="Hubballi", train_type="Express",
             train_number="12627", hour=8, day="Tuesday", weather="Rain", current_delay_min=20),
        dict(current_station="Arsikere", destination="Mysuru", train_type="Passenger",
             train_number="56216", hour=18, day="Friday", weather="Fog", current_delay_min=30),
        dict(current_station="Yesvantpur", destination="Belagavi", train_type="Intercity",
             train_number="16589", hour=9, day="Wednesday", weather="Storm", current_delay_min=50),
        dict(current_station="Kengeri", destination="Mysuru", train_type="MEMU",
             train_number="66553", hour=19, day="Saturday", weather="Clear", current_delay_min=10),
    ]
    rows = []
    for s in scenarios:
        res = predictor.predict(
            current_station=s["current_station"], upcoming_station=None,
            destination=s["destination"], train_type=s["train_type"], hour=s["hour"],
            day=s["day"], weather=s["weather"], current_delay_min=s["current_delay_min"],
        )
        rows.append({
            **s,
            "predicted_delay_min": res["predicted_destination_arrival_delay_min"],
            "alert_level": res["alert"]["level"],
            "top_risk_station": res["alert"]["top_risk_station"],
            "route_km": res["route_km"],
        })
    return {"records": rows}


@app.post("/predict/batch")
def predict_batch(rows: list[BatchRow]):
    out = []
    for r in rows:
        if r.current_station not in predictor.G:
            out.append({"error": f"unknown station {r.current_station}", **r.model_dump()})
            continue
        res = predictor.predict(
            current_station=r.current_station, upcoming_station=r.upcoming_station,
            destination=r.destination, train_type=r.train_type, hour=r.hour,
            day=r.day, weather=r.weather, current_delay_min=r.current_delay_min,
        )
        out.append({
            "current_station": r.current_station,
            "destination": r.destination,
            "predicted_arrival_delay_min": res["predicted_destination_arrival_delay_min"],
            "alert_level": res["alert"]["level"],
            "top_risk_station": res["alert"]["top_risk_station"],
        })
    return out


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse CSV: {e}")

    results = []
    for _, row in df.iterrows():
        station = str(row.get("current_station", "")).strip()
        if station not in predictor.G:
            results.append({"row": int(_), "error": f"unknown station '{station}'"})
            continue
        res = predictor.predict(
            current_station=station,
            upcoming_station=_opt(row, "upcoming_station"),
            destination=_opt(row, "destination"),
            train_type=str(row.get("train_type", "Express")),
            hour=int(row.get("hour", 14)),
            day=str(row.get("day", "Monday")),
            weather=str(row.get("weather", "Clear")),
            current_delay_min=float(row.get("current_delay_min", 0.0)),
        )
        results.append({
            "row": int(_),
            "current_station": station,
            "destination": _opt(row, "destination"),
            "upcoming_station": _opt(row, "upcoming_station"),
            "train_type": str(row.get("train_type", "Express")),
            "train_number": str(row.get("train_number", "")) if pd.notna(row.get("train_number")) else "",
            "hour": int(row.get("hour", 14)),
            "day": str(row.get("day", "Monday")),
            "weather": str(row.get("weather", "Clear")),
            "current_delay_min": float(row.get("current_delay_min", 0.0)),
            "predicted_arrival_delay_min": res["predicted_destination_arrival_delay_min"],
            "alert_level": res["alert"]["level"],
            "top_risk_station": res["alert"]["top_risk_station"],
        })
    df_out = pd.DataFrame(results)
    export_id = f"batch_predictions_{int(time.time())}.csv"
    path = EXPORTS_DIR / export_id
    df_out.to_csv(path, index=False)
    return {"export_id": export_id, "rows": results, "download": f"/export/{export_id}"}


def _opt(row: pd.Series, col: str) -> Optional[str]:
    v = row.get(col)
    if pd.isna(v):
        return None
    return str(v).strip() or None


@app.get("/export/{export_id}")
def export(export_id: str):
    path = EXPORTS_DIR / export_id
    if not path.exists():
        raise HTTPException(404, "export not found")
    content = path.read_text()
    return PlainTextResponse(
        content, media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={export_id}"},
    )


@app.get("/exports")
def list_exports():
    return {"exports": sorted(p.name for p in EXPORTS_DIR.glob("*.csv"))}
