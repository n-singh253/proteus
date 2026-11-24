"""Spotify authorization code flow with PKCE for a user's own development-mode app.

Proteus ships no client ID or secret: each user registers an app in the Spotify developer
dashboard and supplies its client ID. PKCE needs no secret, and the redirect is a loopback
literal (Spotify rejects ``localhost`` and permits plain HTTP only for loopback addresses).
Credentials live in the OS keyring, never in the database, logs or exports.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlsplit

from proteus_music.integrations.http import Transport

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
DEFAULT_REDIRECT = "http://127.0.0.1:8766/callback"
READ_SCOPES = ("user-library-read", "playlist-read-private")
WRITE_SCOPES = ("playlist-modify-private",)  # Proteus creates private playlists only.
TIMEOUT_S = 15.0
EXPIRY_MARGIN_S = 60.0


class AuthorizationError(RuntimeError):
    """Authorization failed or was refused; run `proteus spotify login` again."""


@dataclass(frozen=True)
class Credentials:
    client_id: str
    redirect_uri: str
    access_token: str
    refresh_token: str
    expires_at: float
    scopes: tuple[str, ...]

    def expired(self, now: float) -> bool:
        return now >= self.expires_at - EXPIRY_MARGIN_S


class CredentialStore(Protocol):
    def load(self) -> Credentials | None: ...
    def save(self, credentials: Credentials) -> None: ...
    def delete(self) -> bool: ...


class KeyringStore:
    """One keyring entry holding the user's client ID and tokens as JSON."""

    SERVICE, USER = "proteus-music", "spotify"

    def load(self) -> Credentials | None:
        import keyring

        raw = keyring.get_password(self.SERVICE, self.USER)
        if raw is None:
            return None
        data = json.loads(raw)
        return Credentials(**{**data, "scopes": tuple(data["scopes"])})

    def save(self, credentials: Credentials) -> None:
        import keyring

        keyring.set_password(self.SERVICE, self.USER, json.dumps(asdict(credentials)))

    def delete(self) -> bool:
        import keyring
        from keyring.errors import PasswordDeleteError

        try:
            keyring.delete_password(self.SERVICE, self.USER)
        except PasswordDeleteError:
            return False
        return True


def validate_redirect(uri: str) -> tuple[str, int, str]:
    """(host, port, path) of an explicit-port HTTP loopback redirect URI."""
    parts = urlsplit(uri)
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "::1") or not parts.port:
        raise ValueError(
            "the redirect URI must be http://127.0.0.1:PORT/PATH (or [::1]); Spotify does not "
            "accept 'localhost', and it must match the URI registered for your app exactly"
        )
    return parts.hostname, parts.port, parts.path or "/"


@dataclass(frozen=True)
class AuthorizationRequest:
    url: str
    state: str
    verifier: str


def authorization_request(
    client_id: str, redirect_uri: str, scopes: tuple[str, ...]
) -> AuthorizationRequest:
    verifier = secrets.token_urlsafe(64)  # 86 characters
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    state = secrets.token_urlsafe(24)
    query = urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "state": state,
            "code_challenge_method": "S256",
            "code_challenge": challenge.rstrip(b"=").decode(),
        }
    )
    return AuthorizationRequest(f"{AUTHORIZE_URL}?{query}", state, verifier)


def wait_for_code(redirect_uri: str, state: str, timeout: float = 300.0) -> str:
    """Serve one loopback callback and return its code, checking ``state`` against CSRF."""
    host, port, path = validate_redirect(redirect_uri)
    received: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path != path:
                self.send_error(404)
                return
            received.update({k: v[0] for k, v in parse_qs(url.query).items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Proteus received the Spotify authorization. You can close this tab.")

        def log_message(self, *args: object) -> None:  # Keep the code out of the terminal.
            pass

    server = HTTPServer((host, port), Handler)
    server.timeout = 1.0
    deadline = time.monotonic() + timeout
    try:
        while not received and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if not received:
        raise AuthorizationError("timed out waiting for the Spotify authorization redirect")
    if not secrets.compare_digest(received.get("state", ""), state):
        raise AuthorizationError("the authorization response had the wrong state; try again")
    if "error" in received or "code" not in received:
        raise AuthorizationError(f"Spotify refused authorization: {received.get('error')}")
    return received["code"]


def _token_request(transport: Transport, form: dict[str, str]) -> dict:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = transport.request(
            "POST", TOKEN_URL, headers, urlencode(form).encode(), TIMEOUT_S
        )
    except OSError as error:
        raise AuthorizationError(f"could not reach Spotify accounts: {error}") from error
    if response.status != 200:
        raise AuthorizationError(
            f"Spotify rejected the token request (HTTP {response.status}); "
            "run `proteus spotify login` again"
        )
    return json.loads(response.body)


def _credentials(data: dict, previous: Credentials, now: float) -> Credentials:
    return Credentials(
        client_id=previous.client_id,
        redirect_uri=previous.redirect_uri,
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token") or previous.refresh_token,
        expires_at=now + float(data.get("expires_in", 3600)),
        scopes=tuple(data["scope"].split()) if data.get("scope") else previous.scopes,
    )


def exchange_code(
    transport: Transport,
    client_id: str,
    redirect_uri: str,
    request: AuthorizationRequest,
    code: str,
    clock: Callable[[], float] = time.time,
) -> Credentials:
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": request.verifier,
    }
    blank = Credentials(client_id, redirect_uri, "", "", 0.0, ())
    return _credentials(_token_request(transport, form), blank, clock())


def refresh(
    transport: Transport, credentials: Credentials, clock: Callable[[], float] = time.time
) -> Credentials:
    form = {
        "grant_type": "refresh_token",
        "refresh_token": credentials.refresh_token,
        "client_id": credentials.client_id,
    }
    return _credentials(_token_request(transport, form), credentials, clock())


def open_in_background(url: str) -> None:
    """Open the authorization page without blocking the callback server."""
    import webbrowser

    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
