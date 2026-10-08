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
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audit import AuditMiddleware, log_event, recent as audit_recent
from audit import summary as audit_summary
from auth import (ROLES, Principal, auth_status, authenticate, hash_password,
                  issue_token, require_admin, require_controller, require_dispatcher,
                  require_viewer, throttle, users)
from auth import SESSION_COOKIE, TOKEN_TTL_SECONDS
from config import (ALLOW_SELF_REGISTRATION, ALLOWED_ORIGINS, AUTH_PASS, AUTH_USER,
                    DATA_DIR, EXPORTS_DIR, PROVENANCE_LABELS, RESULTS_DIR, ROOT)
from graph_utils import path_distance_km, shortest_path_km
from inference import CascadePredictor
from station_codes import station_code as _station_code

app = FastAPI(
    title="SWR Cascade Delay API",
    description="Graph-Derived Feature Boosting for Cascading Train Delay Prediction",
    version="2.0.0",
)

# CORS is now an explicit allow-list (was "*"). The dashboard is served from the
# same origin, so cross-origin access is only needed for separately hosted
# clients, which must be named in SWR_ALLOWED_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(AuditMiddleware)


@app.middleware("http")
async def same_origin_cors(request: Request, call_next):
    """Permit credentialed requests that come from this deployment's own origin.

    Sandboxed previews and reverse proxies commonly serve the dashboard from a
    hostname the app cannot know in advance (and an embedded frame can present an
    opaque origin, which makes the browser treat its own API calls as
    cross-origin). The explicit allow-list cannot cover an address that is not
    known at build time, so this middleware reflects the request origin **only
    when its host equals the host the request was addressed to** — a cross-site
    page can never match that, so it grants nothing to a foreign origin.
    """
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")
    same_host = bool(origin) and bool(host) and origin.split("://")[-1] == host
    if same_host and request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": request.headers.get(
                    "access-control-request-headers", "*"),
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
            },
        )
    response = await call_next(request)
    if same_host:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers.append("Vary", "Origin")
    return response

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


predictor = CascadePredictor()

# Dispatcher Copilot — natural-language agent over the live predictor/graph.
from copilot import DispatcherCopilot  # noqa: E402

copilot = DispatcherCopilot(predictor)

# ---------------------------------------------------------------------------
# Users + prediction history
#
# Users now live in a hashed on-disk store (``src/auth.py``) instead of an
# in-memory dict of plaintext passwords, and sessions are signed tokens rather
# than a client-side flag. Prediction history remains in memory: it is a
# convenience view, not a system of record — the audit log is the record.
# ---------------------------------------------------------------------------
HISTORY: list[dict] = []


def _principal_or_none(request: Request) -> Principal | None:
    """Best-effort identity for endpoints that work with or without auth."""
    try:
        return current_user(request)
    except HTTPException:
        return None


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
    """Serve the standalone dispatcher dashboard (proxy/iframe-friendly).

    ``no-store`` matters: the dashboard is the security-relevant client, and a
    cached copy from before an upgrade can keep calling the API the old way
    (unauthenticated), which looks exactly like a broken login.
    """
    return FileResponse(
        str(Path(__file__).parent / "dashboard.html"),
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Dispatcher Copilot (agentic Q&A over the live network)
# ---------------------------------------------------------------------------
@app.get("/copilot/status", dependencies=[Depends(require_dispatcher)])
def copilot_status():
    return copilot.status()


@app.post("/copilot/chat", dependencies=[Depends(require_dispatcher)])
def copilot_chat(req: CopilotRequest):
    return copilot.chat(req.message)


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.dashboard_name}


@app.post("/login")
def login(creds: LoginRequest, request: Request, response: Response):
    """Exchange credentials for a signed session token.

    Brute-force protection: the failure counter is keyed by username *and*
    client IP, and both a lockout and a failed attempt are recorded in the audit
    log. The response never distinguishes "no such user" from "wrong password".

    The token is returned **both** in the response body and as an HttpOnly
    cookie. The body copy is what the dashboard normally uses; the cookie exists
    because some hosting layers and embedded frames strip the ``Authorization``
    header or deny the page access to browser storage, and in that case the
    cookie is the only credential path that survives. Accepting either transport
    costs nothing and removes a whole class of "signed in, but every request is
    401" failures.
    """
    username = (creds.username or "").strip()
    key = f"{username.lower()}|{request.client.host if request.client else 'unknown'}"
    wait = throttle.check(key)
    if wait:
        log_event("login_throttled", username=username, path="/login",
                  status_code=429, detail={"retry_after_s": wait})
        raise HTTPException(
            429, f"Too many failed sign-in attempts. Try again in {wait} seconds.",
            headers={"Retry-After": str(wait)},
        )

    record = authenticate(username, creds.password or "")
    if record is None:
        throttle.record_failure(key)
        log_event("login_failure", username=username, path="/login", status_code=401)
        return {"success": False, "message": "Incorrect username or password."}

    throttle.reset(key)
    token = issue_token(record["username"], record.get("role", "dispatcher"),
                        record.get("full_name", ""))
    log_event("login_success", username=record["username"],
              role=record.get("role"), path="/login", status_code=200)
    # `SameSite=None` is required for the cookie to be sent when the dashboard is
    # embedded in a frame served from another site (an embedded preview, say):
    # a `Lax` cookie is treated as third-party there and never sent, which looks
    # exactly like "logged in, then 401 everywhere". `None` requires `Secure`, so
    # over plain HTTP (local development) we stay with `Lax`, where it still works
    # because everything is same-site.
    # The scheme has to be inferred: behind a proxy the app sees plain HTTP while
    # the browser is on HTTPS. Check the request, the forwarded header, and the
    # browser's own `Origin` — any of them saying https means the connection is
    # secure, and only a Secure cookie may be SameSite=None.
    secure = (request.url.scheme == "https"
              or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"
              or request.headers.get("origin", "").startswith("https://"))
    response.set_cookie(
        SESSION_COOKIE, token["token"],
        max_age=TOKEN_TTL_SECONDS, httponly=True, path="/",
        samesite="none" if secure else "lax",
        secure=secure,
    )
    return {
        "success": True,
        "username": record["username"],
        "full_name": record.get("full_name", ""),
        "role": record.get("role", "dispatcher"),
        "token": token["token"],
        "expires_at": token["expires_at"],
    }


@app.get("/auth/config")
def auth_config():
    """Public: what authentication this deployment expects (no secrets)."""
    return auth_status()


@app.get("/auth/me")
def auth_me(principal: Principal = Depends(require_viewer)):
    """Validate a token and return the caller's identity + permissions."""
    return {
        "username": principal.username, "role": principal.role,
        "full_name": principal.full_name, "expires_at": principal.exp,
        "permissions": {role: principal.has(role) for role in ROLES},
    }


@app.post("/auth/logout")
def auth_logout(response: Response, principal: Principal = Depends(require_viewer)):
    """Tokens are stateless; the client discards it — including the cookie copy."""
    response.delete_cookie(SESSION_COOKIE, path="/")
    log_event("logout", username=principal.username, role=principal.role,
              path="/auth/logout", status_code=200)
    return {"success": True}


@app.post("/register")
def register(req: RegisterRequest):
    """Create a new account (role ``dispatcher``) with a hashed password.

    Self-registration can be disabled entirely with
    ``SWR_ALLOW_SELF_REGISTRATION=0``; when enabled, new accounts are always
    created at the lowest operational role, never as admin.

    Enforces password + email restrictions:
      - password: 8+ chars, at least one uppercase, one lowercase, one digit,
        one special character
      - email: must be a valid address (user@domain.tld)
    """
    if not ALLOW_SELF_REGISTRATION:
        raise HTTPException(403, "Self-registration is disabled on this deployment. "
                                "Ask an administrator to create your account.")
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
    if users.get(u):
        return {"success": False, "message": "Username already exists."}
    try:
        record = users.create(username=u, password_hash=hash_password(req.password),
                              full_name=req.full_name.strip() or u, email=email,
                              role="dispatcher")
    except ValueError as exc:
        return {"success": False, "message": str(exc)}
    # Registration does not sign the user in; they must authenticate normally.
    log_event("user_registered", username=record["username"], role=record["role"],
              path="/register", status_code=200, detail={"email": email})
    return {"success": True, "username": record["username"],
            "full_name": record["full_name"], "role": record["role"],
            "message": "Account created. Sign in to continue."}


@app.get("/system/overview", dependencies=[Depends(require_viewer)])
def system_overview():
    info = predictor.model_info()
    return {
        # --- what is being served, and on what evidence ---------------------
        "provenance": info["provenance"],
        "provenance_label": info["provenance_label"],
        "persistence": info["persistence"],
        "uncertainty": info["uncertainty"],
        "governance": {"audit": audit_summary(), "auth": auth_status()["enabled"]},
        **{k: v for k, v in info.items()
           if k in {"active_model", "available_models", "network_version",
                    "training_rows", "calibration_rows"}},
        **_system_overview_model_block(),
    }


def _system_overview_model_block() -> dict:
    """The model-bake-off block of ``/system/overview`` (unchanged contract)."""
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
    }


@app.get("/graph", dependencies=[Depends(require_viewer)])
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


@app.get("/stations", dependencies=[Depends(require_viewer)])
def stations():
    return {"stations": predictor.station_names()}


@app.get("/stations/{name}", dependencies=[Depends(require_viewer)])
def station_meta(name: str):
    if name not in predictor.G:
        raise HTTPException(404, f"Unknown station '{name}'")
    return predictor.station_meta(name)


@app.get("/model/metrics", dependencies=[Depends(require_viewer)])
def model_metrics():
    return predictor.model_metrics()


@app.get("/models", dependencies=[Depends(require_viewer)])
def models():
    """The selectable model menu (top-5 persisted models) + active model."""
    return {"models": predictor.available_models(),
            "active": predictor.dashboard_name}


@app.post("/model/select", dependencies=[Depends(require_controller)])
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
def predict_manual(req: ManualRequest,
                   principal: Principal = Depends(require_dispatcher)):
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
    _audit_prediction("manual", req.model_dump(), result, principal)
    return result


@app.get("/history", dependencies=[Depends(require_dispatcher)])
def history():
    return {"history": list(reversed(HISTORY))}


# ---------------------------------------------------------------------------
# Live GPS train tracking (reuses the same NetworkX graph + CascadePredictor)
# Flow: GPS (lat,lon) -> nearest station -> current delay -> ML prediction
#       -> NetworkX cascade -> updated risk -> map
# ---------------------------------------------------------------------------
@app.get("/gps/nearest", dependencies=[Depends(require_dispatcher)])
def gps_nearest(lat: float, lon: float):
    """Detect the nearest railway station for a GPS coordinate."""
    name, dist = _nearest_station(lat, lon)
    return {
        "station": name,
        "distance_km": round(dist, 2),
        "lat": predictor.G.nodes[name]["lat"],
        "lon": predictor.G.nodes[name]["lon"],
    }


@app.get("/gps/route", dependencies=[Depends(require_dispatcher)])
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


@app.post("/gps/predict", dependencies=[Depends(require_dispatcher)])
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


@app.get("/demo/records", dependencies=[Depends(require_viewer)])
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


@app.post("/predict/batch", dependencies=[Depends(require_dispatcher)])
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
async def predict_upload(file: UploadFile = File(...),
                         principal: Principal = Depends(require_dispatcher)):
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
    log_event("batch_prediction", path="/predict/upload", status_code=200,
              detail={"export_id": export_id, "rows_scored": int(len(df_out)),
                      "rows_failed": int((df_out.get("error").notna()).sum())
                      if "error" in df_out else 0},
              username=getattr(principal, "username", None),
              role=getattr(principal, "role", None))
    return {"export_id": export_id, "rows": results, "download": f"/export/{export_id}"}


def _opt(row: pd.Series, col: str) -> Optional[str]:
    v = row.get(col)
    if pd.isna(v):
        return None
    return str(v).strip() or None


@app.get("/export/{export_id}", dependencies=[Depends(require_dispatcher)])
def export(export_id: str):
    path = EXPORTS_DIR / export_id
    if not path.exists():
        raise HTTPException(404, "export not found")
    content = path.read_text()
    return PlainTextResponse(
        content, media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={export_id}"},
    )


@app.get("/exports", dependencies=[Depends(require_dispatcher)])
def list_exports():
    return {"exports": sorted(p.name for p in EXPORTS_DIR.glob("*.csv"))}


# ---------------------------------------------------------------------------
# Audit helper used by the prediction endpoints
# ---------------------------------------------------------------------------
def _audit_prediction(kind: str, inputs: dict, result: dict,
                      principal: Optional[Principal] = None) -> None:
    """Record a prediction so it can be reviewed against what actually happened.

    The digest is deliberately compact: the operational inputs (station, delay,
    train type, time) and the headline outputs. Full payloads are not stored, so
    the audit trail cannot become an accidental personal-data store.
    """
    try:
        alert = result.get("alert") or {}
        uncertainty = result.get("uncertainty") or {}
        log_event(
            "prediction",
            username=getattr(principal, "username", None),
            role=getattr(principal, "role", None),
            path=f"/predict/{kind}",
            detail={
                "inputs": {k: inputs.get(k) for k in
                           ("current_station", "upcoming_station", "destination",
                            "train_type", "hour", "day", "weather",
                            "current_delay_min", "train_number")},
                "predicted_delay_min": result.get("predicted_destination_arrival_delay_min"),
                "cascade_level": alert.get("level"),
                "interval_80": (uncertainty.get("intervals") or {}).get("80"),
                "model": result.get("model"),
            },
        )
    except Exception:  # noqa: BLE001 - never block a prediction on auditing
        pass


# ---------------------------------------------------------------------------
# Governance: audit trail + user administration
# ---------------------------------------------------------------------------
@app.get("/admin/audit", dependencies=[Depends(require_admin)])
def admin_audit(limit: int = 100, event: Optional[str] = None,
                username: Optional[str] = None):
    """The append-only audit trail, newest first (admin only)."""
    limit = max(1, min(int(limit), 1000))
    return {"entries": audit_recent(limit, username=username, event=event),
            "summary": audit_summary()}


@app.get("/admin/users", dependencies=[Depends(require_admin)])
def admin_users():
    return {"users": users.all_public(), "roles": list(ROLES)}


class RoleRequest(BaseModel):
    role: str


@app.post("/admin/users/{username}/role", dependencies=[Depends(require_admin)])
def admin_set_role(username: str, req: RoleRequest, principal: Principal = Depends(require_admin)):
    """Change a user's role (admin only). Recorded in the audit trail."""
    try:
        record = users.set_role(username, req.role)
    except KeyError:
        raise HTTPException(404, f"Unknown user '{username}'")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    log_event("role_changed", username=principal.username, role=principal.role,
              path=f"/admin/users/{username}/role", status_code=200,
              detail={"target": username, "new_role": record["role"]})
    return {"success": True, "username": record["username"], "role": record["role"]}


# ---------------------------------------------------------------------------
# Real-data ingestion (the path from simulated to observed traffic)
# ---------------------------------------------------------------------------
@app.post("/ingest/upload", dependencies=[Depends(require_controller)])
async def ingest_upload(file: UploadFile = File(...),
                        principal: Principal = Depends(require_controller)):
    """Ingest an operator export (CSV/Excel) of journey observations.

    Columns are mapped onto the canonical schema automatically where possible;
    anything unmappable is reported rather than silently dropped, and every row
    must pass the data-quality gate before it can influence a model.
    """
    from sources.ingest import ingest_dataframe, read_table

    raw = await file.read()
    tmp_dir = ROOT / "data" / "raw" / "_incoming"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / (file.filename or "upload.csv")
    tmp.write_bytes(raw)
    try:
        df = read_table(tmp)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read '{file.filename}': {exc}")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

    report = ingest_dataframe(df, source_name=file.filename or "upload")
    log_event("ingest", username=principal.username, role=principal.role,
              path="/ingest/upload", status_code=200,
              detail={"source": report.get("source"), "accepted": report.get("accepted"),
                      "rejected": report.get("rejected")})
    return report


class IngestRecordsRequest(BaseModel):
    records: list[dict]
    source: str = "live-feed"


@app.post("/ingest/records", dependencies=[Depends(require_controller)])
def ingest_records(req: IngestRecordsRequest,
                   principal: Principal = Depends(require_controller)):
    """Webhook for a live feed: post canonical journey records as JSON.

    This is the integration point for an authorised NTES/NTES-style feed or a
    control-office export stream. Rows are validated exactly as an upload is.
    """
    from sources.ingest import ingest_dataframe

    if not req.records:
        raise HTTPException(400, "No records supplied.")
    if len(req.records) > 5000:
        raise HTTPException(413, "Send at most 5,000 records per call.")
    frame = pd.DataFrame(req.records)
    report = ingest_dataframe(frame, source_name=req.source)
    log_event("ingest", username=principal.username, role=principal.role,
              path="/ingest/records", status_code=200,
              detail={"source": req.source, "accepted": report.get("accepted"),
                      "rejected": report.get("rejected")})
    return report


@app.get("/ingest/status", dependencies=[Depends(require_viewer)])
def ingest_status():
    """What real data has been ingested, and what that means for provenance."""
    from sources.ingest import detect_provenance, ingestion_history, load_observed

    observed = load_observed()
    provenance = detect_provenance()
    return {
        "provenance": provenance,
        "provenance_label": PROVENANCE_LABELS.get(provenance, provenance),
        "observed_rows": 0 if observed is None else int(len(observed)),
        "history": ingestion_history()[-20:],
        "training_data": predictor.model_info()["provenance"],
        "note": ("Ingested observations are picked up by the next training run "
                 "(`python src/feature_engineering.py && python src/train.py`)."),
    }


# ---------------------------------------------------------------------------
# Data-quality & attribution views
# ---------------------------------------------------------------------------
@app.get("/data/causes", dependencies=[Depends(require_viewer)])
def data_causes():
    """Delay-minutes per cause head — the 'why', not just the 'how late'.

    Empty until real observations with cause codes have been ingested; the
    dashboard shows the explanatory empty state rather than a fabricated chart.
    """
    from sources.causes import CAUSE_HEADS
    from sources.ingest import cause_attribution, load_observed

    rows = cause_attribution()
    return {
        "attribution": rows,
        "taxonomy": [{"cause": k, **v} for k, v in CAUSE_HEADS.items()],
        "observed_rows": 0 if load_observed() is None else int(len(load_observed())),
        "available": bool(rows),
        "note": ("Cause attribution needs ingested observations carrying a cause "
                 "code. Use /ingest/upload with a 'cause' column."),
    }


@app.get("/data/parity", dependencies=[Depends(require_viewer)])
def data_parity():
    """Does the ingested data look like the world the model was trained on?

    Compares the training feature distribution against the ingested observations
    feature-by-feature (two-sample KS). A drift verdict is a statement about
    validity, not accuracy: it says whether the reported metrics can be expected
    to transfer to this traffic at all.
    """
    from sources.ingest import load_observed
    from sources.parity import parity_report

    observed = load_observed()
    if observed is None or not len(observed):
        return {"available": False,
                "note": ("No observed data ingested yet. Upload a real export to "
                         "POST /ingest/upload to run the parity check.")}
    try:
        reference = pd.read_csv(DATA_DIR / "processed" / "features.csv")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Training features unavailable: {exc}")

    shared = [c for c in reference.columns
              if c in observed.columns and pd.api.types.is_numeric_dtype(reference[c])]
    report = parity_report(reference, observed, features=shared)
    report.pop("table", None)          # keep the payload small; the UI shows top rows
    return {"available": True, **report}
