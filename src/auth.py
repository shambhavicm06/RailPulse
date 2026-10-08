"""
Authentication & authorisation for the RailPulse API.

Design goals (why this module exists at all)
--------------------------------------------
The original app gated access **only in the browser**: ``/login`` answered
``{"success": true}`` and ``dashboard.html`` hid the dashboard client-side. Every
API endpoint was therefore reachable by anyone who knew the URL — unacceptable
for a dispatcher tool that can move trains.

This module replaces that with real, dependency-free server-side auth:

* **Password storage** — PBKDF2-HMAC-SHA256 (280k iterations, per-user random
  salt). Plaintext passwords are never stored. Verification is constant-time.
* **Sessions** — stateless, signed tokens (HMAC-SHA256 over a JSON payload) with
  an explicit expiry, so a stolen token is not a permanent key.
* **Roles** — ``viewer < dispatcher < controller < admin``. Each endpoint
  declares the level it needs; the FastAPI dependency enforces it.
* **Brute-force resistance** — per-username/per-IP sliding-window lockout on the
  login endpoint, with generic failure messages (no user enumeration).
* **No hardcoded secret** — the signing key comes from ``SWR_SECRET_KEY`` or a
  generated 0600 file under ``data/``; it is never committed.

Everything here is stdlib, so it adds no supply-chain surface to the project.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from fastapi import Depends, Header, HTTPException, Request, status

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = Path(os.environ.get("SWR_USERS_FILE", DATA_DIR / "users.json"))
SECRET_FILE = DATA_DIR / ".auth_secret"

PBKDF2_ITERATIONS = 280_000
TOKEN_TTL_SECONDS = int(os.environ.get("SWR_TOKEN_TTL", 12 * 3600))  # one shift
#: Cookie name carrying the same signed token. `/login` sets it so that a
#: deployment whose proxy strips `Authorization` (or whose frame denies
#: storage access) still authenticates; the header remains the primary path.
SESSION_COOKIE = "railpulse_token"
LOGIN_MAX_FAILURES = int(os.environ.get("SWR_LOGIN_MAX_FAILURES", 5))
LOGIN_WINDOW_SECONDS = int(os.environ.get("SWR_LOGIN_WINDOW", 300))

# Auth can be disabled for offline demos/tests only; it is ON by default.
AUTH_ENABLED = os.environ.get("SWR_AUTH_DISABLED", "0") != "1"

# Role hierarchy. Higher number = more privilege.
ROLES: dict[str, int] = {"viewer": 0, "dispatcher": 1, "controller": 2, "admin": 3}
DEFAULT_ROLE = "dispatcher"


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------
def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS,
                  salt: Optional[bytes] = None) -> str:
    """Return a self-describing hash: ``pbkdf2_sha256$iters$salt$hash`` (base64)."""
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode(),
        base64.b64encode(dk).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verification of a password against a stored hash."""
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
        return hmac.compare_digest(dk, expected)
    except Exception:  # noqa: BLE001 - malformed record must never authenticate
        return False


# ---------------------------------------------------------------------------
# Signing key
# ---------------------------------------------------------------------------
def _load_or_create_secret() -> bytes:
    env = os.environ.get("SWR_SECRET_KEY")
    if env:
        return env.encode("utf-8")
    if SECRET_FILE.exists():
        raw = SECRET_FILE.read_bytes().strip()
        if raw:
            return raw
    key = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(base64.b64encode(key))
    try:
        SECRET_FILE.chmod(0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return key


_SECRET: bytes = _load_or_create_secret()


# ---------------------------------------------------------------------------
# User store (hashed, on disk)
# ---------------------------------------------------------------------------
class UserStore:
    """A tiny JSON-backed user store.

    Seeded on first boot from ``SWR_USER``/``SWR_PASS`` (default ``admin`` /
    ``swr2026`` for the offline demo, announced loudly in the logs). Passwords
    are only ever persisted as PBKDF2 hashes.
    """

    def __init__(self, path: Path = USERS_FILE, *, seed: bool = True):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._users: dict[str, dict] = {}
        self._load()
        if seed and not self._users:
            self._seed_default_admin()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict):
                self._users = {str(k): v for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            self._users = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._users, indent=2))
        tmp.replace(self.path)
        try:
            self.path.chmod(0o600)
        except OSError:  # pragma: no cover
            pass

    def _seed_default_admin(self) -> None:
        username = os.environ.get("SWR_USER", "admin")
        password = os.environ.get("SWR_PASS", "swr2026")
        role = os.environ.get("SWR_ROLE", "admin")
        default_creds = username == "admin" and password == "swr2026"
        with self._lock:
            self._users[username.lower()] = {
                "username": username,
                "full_name": os.environ.get("SWR_FULL_NAME", "Dispatcher"),
                "email": os.environ.get("SWR_EMAIL", "admin@swr.in"),
                "role": role if role in ROLES else "admin",
                "password_hash": hash_password(password),
                "created_at": time.time(),
                "seeded": True,
            }
            self._save()
        if default_creds:
            print("[auth] ⚠  Seeded the DEFAULT demo account (admin / swr2026). "
                  "Set SWR_USER and SWR_PASS before any real deployment.")

    # -- queries -------------------------------------------------------------
    def get(self, username: str) -> Optional[dict]:
        return self._users.get((username or "").strip().lower())

    def all_public(self) -> list[dict]:
        return [
            {"username": u["username"], "full_name": u.get("full_name", ""),
             "email": u.get("email", ""), "role": u.get("role", DEFAULT_ROLE),
             "created_at": u.get("created_at")}
            for u in self._users.values()
        ]

    # -- mutations -----------------------------------------------------------
    def create(self, *, username: str, password_hash: str, full_name: str,
               email: str, role: str = DEFAULT_ROLE) -> dict:
        key = username.strip().lower()
        with self._lock:
            if key in self._users:
                raise ValueError("Username already exists.")
            record = {
                "username": username.strip(), "full_name": full_name, "email": email,
                "role": role if role in ROLES else DEFAULT_ROLE,
                "password_hash": password_hash, "created_at": time.time(),
            }
            self._users[key] = record
            self._save()
            return record

    def set_role(self, username: str, role: str) -> dict:
        key = username.strip().lower()
        if role not in ROLES:
            raise ValueError(f"Unknown role '{role}'. One of: {', '.join(ROLES)}")
        with self._lock:
            rec = self._users.get(key)
            if rec is None:
                raise KeyError(username)
            rec["role"] = role
            self._save()
            return rec


users = UserStore()


# ---------------------------------------------------------------------------
# Login throttling
# ---------------------------------------------------------------------------
class LoginThrottle:
    """Sliding-window failure counter keyed by ``username|client-ip``."""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES,
                 window: int = LOGIN_WINDOW_SECONDS):
        self.max_failures = max_failures
        self.window = window
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> list[float]:
        hits = [t for t in self._hits.get(key, []) if now - t < self.window]
        self._hits[key] = hits
        return hits

    def check(self, key: str) -> int:
        """Return seconds to wait, or 0 if the caller may attempt a login."""
        now = time.time()
        with self._lock:
            hits = self._prune(key, now)
            if len(hits) >= self.max_failures:
                return max(1, int(self.window - (now - hits[0])))
        return 0

    def record_failure(self, key: str) -> None:
        now = time.time()
        with self._lock:
            hits = self._prune(key, now)
            hits.append(now)
            self._hits[key] = hits

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


throttle = LoginThrottle()


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""
    username: str
    role: str
    full_name: str = ""
    exp: int = 0

    @property
    def level(self) -> int:
        return ROLES.get(self.role, 0)

    def has(self, required: str) -> bool:
        return self.level >= ROLES.get(required, 0)


def issue_token(username: str, role: str, full_name: str = "",
                ttl: int = TOKEN_TTL_SECONDS) -> dict:
    """Mint a signed session token. Returns the token and its expiry."""
    now = int(time.time())
    payload = {
        "sub": username, "role": role, "name": full_name,
        "iat": now, "exp": now + ttl, "jti": secrets.token_hex(8),
    }
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(_SECRET, body.encode(), hashlib.sha256).digest())
    return {"token": f"v1.{body}.{sig}", "expires_at": payload["exp"], "payload": payload}


def decode_token(token: str) -> Optional[Principal]:
    """Verify signature + expiry. Returns ``None`` for any invalid token."""
    try:
        version, body, sig = token.split(".")
        if version != "v1":
            return None
        expected = _b64e(hmac.new(_SECRET, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64d(body))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return Principal(username=payload.get("sub", ""),
                         role=payload.get("role", "viewer"),
                         full_name=payload.get("name", ""),
                         exp=int(payload.get("exp", 0)))
    except Exception:  # noqa: BLE001 - any parse/verify failure is unauthorised
        return None


def authenticate(username: str, password: str) -> Optional[dict]:
    """Check credentials against the store. Returns the record on success."""
    rec = users.get(username)
    if not rec:
        # Spend a comparable amount of time so a missing user is not
        # distinguishable from a wrong password (no user enumeration).
        verify_password(password, hash_password("dummy") )
        return None
    if not verify_password(password, rec.get("password_hash", "")):
        return None
    return rec


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------
def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


#: A non-standard header carrying the same signed token. Some hosting layers
#: strip ``Authorization`` (it can leak credentials to upstream services) while
#: leaving unknown headers alone, so the dashboard sends both and the server
#: accepts either.
TOKEN_HEADER = "x-railpulse-token"


def _token_carriers(request: Request,
                    authorization: Optional[str]) -> dict:
    """Every place a token may arrive from, most reliable first.

    A browser in a third-party frame cannot be relied on for either standard
    carrier: ``Authorization`` may be stripped by a proxy middleware, and a
    session cookie is refused outright when it counts as third-party. Accepting
    several carriers costs nothing and makes the deployment work wherever it is
    hosted; the token itself is identical in all of them.
    """
    custom = request.headers.get(TOKEN_HEADER)
    if custom and custom.lower().startswith("bearer "):
        custom = custom.split(None, 1)[1]
    query = request.query_params.get("token") or request.query_params.get("_t")
    ordered = (
        ("header", _bearer(authorization)),
        ("custom_header", custom),
        ("cookie", request.cookies.get(SESSION_COOKIE)),
        ("query", query),
    )
    for source, value in ordered:
        if value:
            return {"source": source, "token": value}
    return {"source": "none", "token": None}


def current_user(request: Request,
                 authorization: Optional[str] = Header(default=None)) -> Principal:
    """Resolve the caller. Public endpoints may still want the identity."""
    if not AUTH_ENABLED:
        return Principal(username="anonymous-auth-disabled", role="admin",
                         full_name="Auth disabled")
    carriers = _token_carriers(request, authorization)
    token = carriers["token"]
    principal = decode_token(token) if token else None
    if principal is None:
        # Say *why*: "no credential arrived at all" and "a credential arrived but
        # was rejected" look identical to a user, yet they have completely
        # different causes (a proxy stripping headers or a frame blocking storage,
        # versus a genuinely expired session). The reason is also recorded in the
        # audit trail so a failure can be diagnosed after the fact.
        source = carriers["source"]
        detail = {
            "none": "No session credential arrived with this request. The token is "
                    "sent three ways (Authorization header, X-RailPulse-Token "
                    "header, session cookie); none of them reached the server, so "
                    "the hosting layer is dropping them. Open the app in its own "
                    "browser tab — a frame is the usual cause.",
            "header": "The session token in the Authorization header is invalid or "
                      "expired.",
            "custom_header": "The session token in the X-RailPulse-Token header is "
                             "invalid or expired.",
            "cookie": "The session cookie is invalid or expired.",
            "query": "The session token in the URL is invalid or expired.",
        }[source]
        try:
            from audit import log_event
            log_event("auth_rejected", path=str(request.url.path),
                      status_code=401, detail={"credential": source})
        except Exception:  # noqa: BLE001 - auditing must never mask the 401
            pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Expose the identity to the audit middleware (same request scope).
    try:
        request.state.principal = principal
    except Exception:  # noqa: BLE001 - state is best-effort
        pass
    return principal


def require_role(required: str):
    """Dependency factory: enforce a minimum role for an endpoint."""

    def _dep(request: Request,
             authorization: Optional[str] = Header(default=None)) -> Principal:
        principal = current_user(request, authorization)
        if not principal.has(required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role '{required}' or higher; "
                       f"you are '{principal.role}'.",
            )
        return principal

    return _dep


# Convenience dependencies used by app.py (one per role level)
require_viewer = require_role("viewer")
require_dispatcher = require_role("dispatcher")
require_controller = require_role("controller")
require_admin = require_role("admin")


def auth_status() -> dict:
    """Non-secret summary of the auth configuration (for /auth/config)."""
    seeded_default = any(
        u.get("seeded") and u.get("username", "").lower() == "admin"
        for u in users._users.values()  # noqa: SLF001 - internal summary only
    )
    return {
        "enabled": AUTH_ENABLED,
        "roles": list(ROLES),
        "token_ttl_seconds": TOKEN_TTL_SECONDS,
        "login_max_failures": LOGIN_MAX_FAILURES,
        "login_window_seconds": LOGIN_WINDOW_SECONDS,
        "password_hashing": f"pbkdf2_sha256 ({PBKDF2_ITERATIONS} iterations)",
        "user_store": str(users.path.relative_to(ROOT)) if users.path.is_relative_to(ROOT)
                      else str(users.path),
        "default_demo_account_active": seeded_default,
        "warning": ("Change the default demo credentials (SWR_USER/SWR_PASS) before "
                    "deploying." if seeded_default else None),
    }
