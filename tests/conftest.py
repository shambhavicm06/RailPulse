"""
Shared pytest fixtures.

Design notes
------------
* Tests are **hermetic**: the user store, audit database and secret file are
  redirected into a temporary directory, so running the suite never touches the
  developer's data or the repository's real audit trail.
* The API tests use FastAPI's ``TestClient`` against the real app object, so the
  dependency wiring, middleware and role checks under test are exactly those that
  serve production traffic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def isolated_state(tmp_path_factory):
    """Point auth + audit state at a temporary directory for the whole session."""
    tmp = tmp_path_factory.mktemp("railpulse-state")
    os.environ.setdefault("SWR_USERS_FILE", str(tmp / "users.json"))
    os.environ.setdefault("SWR_AUDIT_DB", str(tmp / "audit.db"))
    os.environ.setdefault("SWR_SECRET_KEY", "test-secret-key-not-for-production")
    yield tmp


@pytest.fixture(scope="session")
def app_module(isolated_state):
    import app  # noqa: PLC0415 - imported after env vars are set
    return app


@pytest.fixture(scope="session")
def client(app_module):
    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def admin_token(client, app_module) -> str:
    """A valid admin session token (the seeded demo account)."""
    from config import AUTH_PASS, AUTH_USER
    response = client.post("/login", json={"username": AUTH_USER, "password": AUTH_PASS})
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    return response.json()["token"]


@pytest.fixture(scope="session")
def admin_headers(admin_token) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="session")
def dispatcher_headers(client, app_module) -> dict:
    """A low-privilege account, created through the real store (not a mock)."""
    from auth import hash_password, issue_token, users

    username = "test-dispatcher"
    if not users.get(username):
        users.create(username=username, password_hash=hash_password("Dispatcher#1"),
                     full_name="Test Dispatcher", email="dispatcher@swr.in",
                     role="dispatcher")
    token = issue_token(username, "dispatcher", "Test Dispatcher")
    return {"Authorization": f"Bearer {token['token']}"}


@pytest.fixture(autouse=True)
def isolated_ingestion(tmp_path, monkeypatch):
    """Keep ingested observations out of the repository.

    Without this, running the suite appends test rows to
    ``data/raw/observed_journeys.csv`` — which both pollutes the training data and
    makes the tests order-dependent (a second run sees its own rows as duplicates).
    """
    import sources.ingest as ingest

    monkeypatch.setattr(ingest, "OBSERVED_CSV", tmp_path / "observed_journeys.csv")
    monkeypatch.setattr(ingest, "INGEST_LOG", tmp_path / "ingestion_log.json")
    yield


@pytest.fixture(autouse=True)
def no_ambient_credentials(client):
    """Each test starts and ends with an empty cookie jar.

    The HTTP client is session-scoped, and cookie authentication means a sign-in
    in one test would otherwise leave every later test authenticated — turning
    "anonymous callers are rejected" assertions green for the wrong reason.
    """
    client.cookies.clear()
    yield
    client.cookies.clear()
