"""
Authentication unit tests.

These cover the properties a security review would ask about: passwords are
never stored in the clear, tokens cannot be forged or outlive their expiry,
brute-force attempts are throttled, and the role hierarchy is enforced in the
right direction.
"""
from __future__ import annotations

import time

import pytest

import auth


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def test_password_is_hashed_not_stored():
    hashed = auth.hash_password("swr2026")
    assert "swr2026" not in hashed
    assert hashed.startswith("pbkdf2_sha256$")
    assert auth.verify_password("swr2026", hashed) is True


def test_password_verification_rejects_wrong_password():
    hashed = auth.hash_password("correct horse battery staple")
    assert auth.verify_password("Correct horse battery staple", hashed) is False
    assert auth.verify_password("", hashed) is False


def test_salt_makes_identical_passwords_hash_differently():
    assert auth.hash_password("same-password") != auth.hash_password("same-password")


def test_malformed_hash_never_authenticates():
    for bad in ("", "not-a-hash", "pbkdf2_sha256$only$three", "md5$1$a$b"):
        assert auth.verify_password("anything", bad) is False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def test_token_roundtrip_carries_identity():
    issued = auth.issue_token("dispatcher-1", "dispatcher", "Dee")
    principal = auth.decode_token(issued["token"])
    assert principal is not None
    assert principal.username == "dispatcher-1"
    assert principal.role == "dispatcher"
    assert principal.full_name == "Dee"


def test_tampered_token_is_rejected():
    token = auth.issue_token("admin", "admin")["token"]
    header, body, signature = token.split(".")
    assert auth.decode_token(f"{header}.{body}.{signature[:-4]}AAAA") is None
    assert auth.decode_token("v1.garbage.signature") is None
    assert auth.decode_token("") is None
    assert auth.decode_token("not-even-a-token") is None


def test_token_payload_cannot_be_escalated_without_the_key():
    """Swapping the role in the payload must invalidate the signature."""
    import base64
    import json

    token = auth.issue_token("low-privilege", "viewer")["token"]
    header, body, signature = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["role"] = "admin"
    forged_body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    assert auth.decode_token(f"{header}.{forged_body}.{signature}") is None


def test_expired_token_is_rejected():
    issued = auth.issue_token("admin", "admin", ttl=-1)
    assert auth.decode_token(issued["token"]) is None


def test_role_hierarchy_is_ordered():
    viewer = auth.Principal("v", "viewer")
    controller = auth.Principal("c", "controller")
    admin = auth.Principal("a", "admin")
    assert not viewer.has("dispatcher")
    assert controller.has("dispatcher") and controller.has("viewer")
    assert not controller.has("admin")
    assert admin.has("admin") and admin.has("dispatcher")


# ---------------------------------------------------------------------------
# Login throttling
# ---------------------------------------------------------------------------
def test_throttle_blocks_after_repeated_failures_and_recovers():
    throttle = auth.LoginThrottle(max_failures=3, window=1)
    key = "user|127.0.0.1"
    assert throttle.check(key) == 0
    for _ in range(3):
        throttle.record_failure(key)
    assert throttle.check(key) > 0          # locked out
    time.sleep(1.05)
    assert throttle.check(key) == 0         # window expired
    throttle.record_failure(key)
    throttle.reset(key)
    assert throttle.check(key) == 0         # successful login clears the counter


def test_authenticate_uses_hashed_store():
    from auth import authenticate, hash_password, users

    username = "unit-test-user"
    if not users.get(username):
        users.create(username=username, password_hash=hash_password("Passw0rd!x"),
                     full_name="Unit Test", email="unit@swr.in", role="viewer")
    assert authenticate(username, "Passw0rd!x") is not None
    assert authenticate(username, "wrong") is None
    assert authenticate("no-such-user", "whatever") is None
    # The stored record must never contain the plaintext.
    record = users.get(username)
    assert "Passw0rd!x" not in str(record)


def test_seeded_demo_account_exists_for_offline_use():
    record = auth.users.get("admin")
    assert record is not None
    assert record["password_hash"].startswith("pbkdf2_sha256$")
    assert auth.auth_status()["default_demo_account_active"] is True
