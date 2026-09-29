from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

SESSION_COOKIE_NAME = "code_intelligence_session"
SESSION_LIFETIME_SECONDS = 8 * 60 * 60
_MAX_SESSION_TOKEN_BYTES = 2_048
_MAX_CLOCK_SKEW_SECONDS = 60
_SIGNING_CONTEXT = b"ai-code-intelligence/browser-session/v1"


@dataclass(frozen=True, slots=True)
class BrowserSession:
    """Verified, short-lived browser session values carried in a signed cookie."""

    csrf_token: str
    expires_at: datetime


class BrowserSessionManager:
    """Issue and verify stateless browser sessions without persisting the API key."""

    def __init__(
        self,
        api_key: str,
        *,
        lifetime_seconds: int = SESSION_LIFETIME_SECONDS,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if lifetime_seconds < 60:
            raise ValueError("session lifetime must be at least 60 seconds")
        self._lifetime_seconds = lifetime_seconds
        self._api_key_digest = hashlib.sha256(api_key.encode("utf-8")).digest()
        self._signing_key = hmac.digest(api_key.encode("utf-8"), _SIGNING_CONTEXT, "sha256")

    def valid_api_key(self, supplied: str) -> bool:
        """Compare a supplied API key in constant time."""

        supplied_digest = hashlib.sha256(supplied.encode("utf-8")).digest()
        return hmac.compare_digest(self._api_key_digest, supplied_digest)

    def issue(self, *, now: int | None = None) -> tuple[str, BrowserSession]:
        """Return a signed cookie token and its browser-visible session values."""

        issued_at = int(time.time()) if now is None else now
        expires_at = issued_at + self._lifetime_seconds
        csrf_token = secrets.token_urlsafe(32)
        payload = {
            "csrf": csrf_token,
            "exp": expires_at,
            "iat": issued_at,
            "v": 1,
        }
        encoded_payload = _encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signature = _encode(hmac.digest(self._signing_key, encoded_payload.encode("ascii"), "sha256"))
        return (
            f"{encoded_payload}.{signature}",
            BrowserSession(
                csrf_token=csrf_token,
                expires_at=datetime.fromtimestamp(expires_at, tz=UTC),
            ),
        )

    def verify(self, token: str | None, *, now: int | None = None) -> BrowserSession | None:
        """Verify a cookie signature, lifetime, and payload shape; return None on any failure."""

        if not token:
            return None
        try:
            token_bytes = token.encode("ascii")
        except UnicodeEncodeError:
            return None
        if len(token_bytes) > _MAX_SESSION_TOKEN_BYTES:
            return None
        encoded_payload, separator, supplied_signature = token.partition(".")
        if not separator or not encoded_payload or not supplied_signature or "." in supplied_signature:
            return None
        expected_signature = _encode(
            hmac.digest(self._signing_key, encoded_payload.encode("ascii"), "sha256")
        )
        if not hmac.compare_digest(expected_signature, supplied_signature):
            return None
        try:
            payload = json.loads(_decode(encoded_payload))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or set(payload) != {"csrf", "exp", "iat", "v"}:
            return None
        csrf_token = payload.get("csrf")
        expires_at = payload.get("exp")
        issued_at = payload.get("iat")
        version = payload.get("v")
        if (
            not isinstance(csrf_token, str)
            or len(csrf_token) < 32
            or type(expires_at) is not int
            or type(issued_at) is not int
            or type(version) is not int
            or version != 1
        ):
            return None
        current_time = int(time.time()) if now is None else now
        lifetime = expires_at - issued_at
        if (
            expires_at <= current_time
            or issued_at > current_time + _MAX_CLOCK_SKEW_SECONDS
            or lifetime <= 0
            or lifetime > self._lifetime_seconds
        ):
            return None
        return BrowserSession(
            csrf_token=csrf_token,
            expires_at=datetime.fromtimestamp(expires_at, tz=UTC),
        )


class LoginRateLimiter:
    """Bound failed-login pressure with a small in-process, per-client sliding window."""

    def __init__(
        self,
        *,
        max_attempts: int = 8,
        window_seconds: float = 60,
        max_clients: int = 4_096,
    ) -> None:
        if min(max_attempts, window_seconds, max_clients) <= 0:
            raise ValueError("rate-limiter bounds must be positive")
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._max_clients = max_clients
        self._attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()

    def allow(self, client_id: str, *, now: float | None = None) -> bool:
        """Record one attempt and return whether it is within the configured limit."""

        current_time = time.monotonic() if now is None else now
        cutoff = current_time - self._window_seconds
        with self._lock:
            attempts = self._attempts.pop(client_id, deque())
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            allowed = len(attempts) < self._max_attempts
            attempts.append(current_time)
            self._attempts[client_id] = attempts
            while len(self._attempts) > self._max_clients:
                self._attempts.popitem(last=False)
            return allowed


def csrf_matches(session: BrowserSession, supplied: str) -> bool:
    """Compare a browser CSRF header with the token bound into its signed session."""

    return bool(supplied) and hmac.compare_digest(
        session.csrf_token.encode("utf-8"),
        supplied.encode("utf-8"),
    )


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    return decoded.decode("utf-8")
