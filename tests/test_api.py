"""
API contract and authorisation tests.

The central claim being tested: **the API is no longer open**. Every operational
endpoint rejects anonymous callers, role requirements are enforced server-side
(not just hidden in the browser), and the audit trail records what happened.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from auth import SESSION_COOKIE

# Endpoints that must stay public: the dashboard shell, health probes and the
# sign-in flow itself. Everything else requires a token.
PUBLIC_PATHS = ["/health", "/auth/config"]

# (method, path) pairs that must reject anonymous callers with 401.
PROTECTED_ENDPOINTS = [
    ("GET", "/graph"),
    ("GET", "/stations"),
    ("GET", "/stations/Mysuru"),
    ("GET", "/system/overview"),
    ("GET", "/models"),
    ("GET", "/model/metrics"),
    ("GET", "/history"),
    ("GET", "/exports"),
    ("GET", "/demo/records"),
    ("GET", "/copilot/status"),
    ("GET", "/ingest/status"),
    ("GET", "/data/causes"),
    ("GET", "/data/parity"),
    ("POST", "/predict/manual"),
    ("POST", "/predict/batch"),
    ("POST", "/model/select"),
    ("POST", "/copilot/chat"),
    ("POST", "/ingest/records"),
    ("GET", "/admin/audit"),
    ("GET", "/admin/users"),
]


def _call(client, method: str, path: str, headers=None, json=None):
    if method == "GET":
        return client.get(path, headers=headers or {})
    return client.post(path, headers=headers or {}, json=json or {})


@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_public_endpoints_need_no_token(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
def test_protected_endpoints_reject_anonymous(client, method, path):
    response = _call(client, method, path)
    assert response.status_code == 401, f"{method} {path} -> {response.status_code}"
    assert "credential" in response.json()["detail"].lower()


def test_garbage_token_is_rejected(client):
    headers = {"Authorization": "Bearer v1.bm90.a-token"}
    assert client.get("/graph", headers=headers).status_code == 401
    assert client.get("/graph", headers={"Authorization": "Basic abc"}).status_code == 401


def test_login_returns_a_usable_token(client, admin_headers):
    me = client.get("/auth/me", headers=admin_headers)
    assert me.status_code == 200
    body = me.json()
    assert body["username"] == "admin"
    assert body["permissions"]["admin"] is True


def test_login_failure_is_generic_and_constant_shaped(client):
    response = client.post("/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 200
    assert response.json()["success"] is False
    # No user enumeration: an unknown username answers exactly like a bad password.
    unknown = client.post("/login", json={"username": "ghost", "password": "wrong"})
    assert unknown.json() == response.json()


def test_admin_endpoints_reject_low_privilege_users(client, dispatcher_headers):
    """A dispatcher must not read the audit trail or change roles."""
    assert client.get("/admin/audit", headers=dispatcher_headers).status_code == 403
    assert client.get("/admin/users", headers=dispatcher_headers).status_code == 403


def test_model_switch_requires_controller_role(client, dispatcher_headers):
    response = client.post("/model/select", json={"model": "LightGBM"},
                           headers=dispatcher_headers)
    assert response.status_code == 403


def test_dispatcher_can_predict_but_not_administer(client, dispatcher_headers):
    response = client.post("/predict/manual", headers=dispatcher_headers,
                           json={"current_station": "Mysuru", "destination": "Hubballi",
                                 "current_delay_min": 20, "hour": 9, "day": "Monday",
                                 "weather": "Clear", "train_type": "Express"})
    assert response.status_code == 200, response.text
    assert "predicted_destination_arrival_delay_min" in response.json()
    assert client.get("/admin/audit", headers=dispatcher_headers).status_code == 403


# ---------------------------------------------------------------------------
# Functional contract
# ---------------------------------------------------------------------------
def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"]


def test_prediction_returns_interval_and_probabilities(client, admin_headers):
    response = client.post("/predict/manual", headers=admin_headers,
                           json={"current_station": "KSR Bengaluru",
                                 "destination": "Mysuru", "current_delay_min": 25,
                                 "hour": 9, "day": "Monday", "weather": "Rain",
                                 "train_type": "Express"})
    assert response.status_code == 200
    body = response.json()
    point = body["predicted_destination_arrival_delay_min"]
    assert isinstance(point, (int, float)) and point >= 0

    uncertainty = body.get("uncertainty", {})
    if uncertainty.get("available"):
        for band in uncertainty["intervals"].values():
            # The headline number must never fall outside its own band.
            assert band["lower"] <= point <= band["upper"]
            assert 0 <= band["lower"] <= band["upper"]
        for key, probability in uncertainty["exceedance"].items():
            assert key.startswith("p_gt_")
            assert 0.0 <= probability <= 1.0
    else:
        pytest.skip("bundle has no quantile models (run src/train.py to add them)")


def test_unknown_station_is_a_404(client, admin_headers):
    response = client.post("/predict/manual", headers=admin_headers,
                           json={"current_station": "Atlantis Central"})
    assert response.status_code == 404


def test_overview_discloses_provenance_and_persistence(client, admin_headers):
    body = client.get("/system/overview", headers=admin_headers).json()
    assert body["provenance"] in {"synthetic", "synthetic+real", "real"}
    assert body["provenance_label"]
    assert "portable_formats" in body["persistence"]
    assert "enabled" in body["uncertainty"]
    assert body["governance"]["auth"] is True
    assert body["n_stations"] > 0 and body["best_mae"] >= 0


def test_ingest_rejects_unmappable_payload(client, admin_headers):
    response = client.post("/ingest/records", headers=admin_headers,
                           json={"records": [{"nonsense": 1}], "source": "unit-test"})
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 0
    assert "error" in body or body["rejected"] == 1


def test_ingest_accepts_a_realistic_export(client, admin_headers):
    """An operator-shaped export must map, validate and be stored."""
    records = [
        {"Train No": "12627", "Type": "SF Exp", "From Station": "KSR Bengaluru",
         "Station": "Mysuru", "To Station": "Dharwad", "Sch Time": "14:05",
         "Day": "Mon", "Wx": "fog", "Delay": 25, "Arrival Delay": 55,
         "cause": "preoccupied line"},
        {"Train No": "2244", "Type": "Express", "From Station": "MYS",
         "Station": "Arsikere", "To Station": "Chennai Central", "Sch Time": "7 PM",
         "Day": "sat", "Wx": "Clear", "Delay": 90, "Arrival Delay": 130,
         "cause": "loco failure"},
    ]
    body = client.post("/ingest/records", headers=admin_headers,
                       json={"records": records, "source": "pytest-export"}).json()
    assert body["accepted"] == 2, body
    # Cause heads must be classified, and the report must state what was derived.
    assert body["normalisation"]["rows_out"] == 2

    causes = client.get("/data/causes", headers=admin_headers).json()
    heads = {row["cause"] for row in causes["attribution"]}
    assert {"PRE_OCCUPIED_LINE", "ROLLING_STOCK"} <= heads

    parity = client.get("/data/parity", headers=admin_headers).json()
    assert parity["available"] is True
    assert parity["verdict"] in {"compatible", "distribution_shift_detected"}


def test_ingest_refuses_a_feed_with_no_label(client, admin_headers):
    """A live position feed is not training data — it has no supervised label.

    The old normaliser filled a missing delay column with ``0.0``, i.e. it would
    have appended "arrived exactly on time" rows and trained on them. A batch with
    no label must be refused, with a message that says what to do instead.
    """
    body = client.post("/ingest/records", headers=admin_headers,
                       json={"records": [{"Station": "Mysuru", "Delay": 25}],
                             "source": "live-position-feed"}).json()
    assert body["accepted"] == 0
    assert "destination_arrival_delay_min" in body["error"]
    assert "/predict" in body["error"]

    # A row that claims to be arriving where it already is cannot be trusted.
    body = client.post("/ingest/records", headers=admin_headers,
                       json={"records": [{"Station": "Mysuru", "To Station": "Mysuru",
                                          "Delay": 10, "Arrival Delay": 20}],
                             "source": "self-destination"}).json()
    assert body["accepted"] == 0
    assert body["validation"]["checks"]["placement"]["rejected"] == 1

    # A "next station" that is not adjacent to the current one resolves to no
    # section: rejected rather than silently handed to the model as a 0 km hop.
    body = client.post("/ingest/records", headers=admin_headers,
                       json={"records": [{"Station": "Mysuru", "Next Station": "Dharwad",
                                          "To Station": "Dharwad",
                                          "Delay": 10, "Arrival Delay": 20}],
                             "source": "non-adjacent-hop"}).json()
    assert body["accepted"] == 0
    assert body["validation"]["checks"]["section_plausibility"]["rejected"] == 1


def test_audit_trail_records_predictions_and_logins(client, admin_headers):
    # Sign in through the endpoint (tokens in the fixtures are minted directly),
    # then predict, so the trail is exercised rather than assumed.
    client.post("/login", json={"username": "admin", "password": "swr2026"})
    client.post("/predict/manual", headers=admin_headers,
                json={"current_station": "Mysuru", "destination": "KSR Bengaluru",
                      "current_delay_min": 30})
    body = client.get("/admin/audit?limit=200", headers=admin_headers).json()
    events = {entry["event"] for entry in body["entries"]}
    assert "login_success" in events
    assert "prediction" in events
    assert body["summary"]["successful_logins"] >= 1
    prediction_rows = [e for e in body["entries"] if e["event"] == "prediction"]
    assert prediction_rows and prediction_rows[0]["username"]
    assert prediction_rows[0]["detail"]


def test_audit_trail_records_denied_attempts(client, dispatcher_headers, admin_headers):
    client.get("/admin/audit", headers=dispatcher_headers)      # 403 → logged
    body = client.get("/admin/audit?event=auth_denied&limit=50",
                      headers=admin_headers).json()
    assert body["summary"]["auth_denied"] >= 1


def test_tiles_and_static_are_public(client):
    """Map tiles and figures must load before sign-in, or the login page breaks."""
    assert client.get("/static/model_metrics.csv").status_code == 200
    assert client.get("/lib/leaflet.js").status_code == 200


# ---------------------------------------------------------------------------
# Credential transports
#
# A "signed in, but every request is 401" failure has exactly two causes: the
# Authorization header never reached the server (proxy stripping it), or the page
# could not keep the token (blocked storage). Both must keep working, so the
# token travels over two independent transports and each is tested alone.
# ---------------------------------------------------------------------------
def test_bearer_header_alone_authenticates(client):
    body = client.post("/login", json={"username": "admin", "password": "swr2026"}).json()
    headers = {"Authorization": f"Bearer {body['token']}"}
    assert client.get("/system/overview", headers=headers).status_code == 200


def test_session_cookie_alone_authenticates(client, app_module):
    """Simulate the header being stripped: only the cookie may be sent."""
    login = client.post("/login", json={"username": "admin", "password": "swr2026"})
    assert login.status_code == 200
    assert SESSION_COOKIE in login.cookies or SESSION_COOKIE in client.cookies

    # A fresh client that carries only the cookie — no Authorization header.
    cookie_only = TestClient(app_module.app)
    cookie_only.cookies.set(SESSION_COOKIE, login.cookies[SESSION_COOKIE])
    assert cookie_only.get("/system/overview").status_code == 200
    assert cookie_only.post("/predict/manual", json={
        "current_station": "Mysuru", "destination": "KSR Bengaluru",
        "current_delay_min": 20}).status_code == 200
    # ...and the role gate still applies to it.
    assert cookie_only.get("/admin/audit").status_code == 200   # admin role


def test_no_credential_says_so(client):
    body = client.get("/system/overview")
    assert body.status_code == 401
    assert "credential" in body.json()["detail"].lower()


def test_rejected_credential_says_which_transport(client):
    body = client.get("/system/overview", headers={"Authorization": "Bearer not-a-token"})
    assert body.status_code == 401
    assert "Authorization header" in body.json()["detail"]


def test_logout_clears_the_cookie(client):
    client.post("/login", json={"username": "admin", "password": "swr2026"})
    headers = {"Authorization": f"Bearer "
               f"{client.post('/login', json={'username': 'admin', 'password': 'swr2026'}).json()['token']}"}
    assert client.post("/auth/logout", headers=headers).status_code == 200
    assert SESSION_COOKIE not in client.cookies


def test_same_host_preflight_is_answered_with_credentials(client):
    """A frame that sees its own API calls as cross-origin must still work.

    Only an origin whose host equals the addressed host is reflected, so this
    grants nothing to a foreign site.
    """
    r = client.options("/predict/manual", headers={
        "Origin": "http://testserver", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type"})
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == "http://testserver"
    assert r.headers["access-control-allow-credentials"] == "true"


def test_foreign_origin_preflight_is_not_reflected(client):
    r = client.options("/predict/manual", headers={
        "Origin": "https://evil.example.com",
        "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") in (None, "")
