"""Spotify Web API client: token refresh, pagination, rate limits and bounded retries.

Endpoint paths follow the February 2026 development-mode changes (``/me/playlists`` to
create, ``/playlists/{id}/items`` to read and add). A 429 carrying ``QUOTA_EXCEEDED`` means
the development-mode quota is spent and is not retried; other 429s honour Retry-After.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlencode

from proteus_music.integrations.http import HttpResponse, Transport
from proteus_music.integrations.spotify import auth
from proteus_music.integrations.spotify.auth import Credentials, CredentialStore

API_ROOT = "https://api.spotify.com/v1"
PAGE_SIZE = 50  # Documented maximum for library, playlist and item listings.
ADD_BATCH = 100  # Documented maximum items per add request.
SEARCH_LIMIT = 10  # Development-mode maximum since February 2026.
TIMEOUT_S = 20.0
BACKOFF_S = (1.0, 2.0, 4.0, 8.0)  # One entry per retry.
MAX_RETRY_AFTER_S = 120.0
_RETRYABLE = {500, 502, 503, 504}


class SpotifyError(RuntimeError):
    """A request failed for a reason retrying will not fix."""


class RateLimitedError(SpotifyError):
    """Spotify is throttling or the quota is spent; completed work is kept, re-run later."""


class SpotifyClient:
    def __init__(
        self,
        transport: Transport,
        store: CredentialStore,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        credentials = store.load()
        if credentials is None:
            raise auth.AuthorizationError("not connected; run `proteus spotify login` first")
        self.transport, self.store, self.sleep, self.clock = transport, store, sleep, clock
        self.credentials: Credentials = credentials
        self.requests = 0

    def has_scopes(self, scopes: tuple[str, ...]) -> bool:
        return set(scopes) <= set(self.credentials.scopes)

    def _refresh(self) -> None:
        self.credentials = auth.refresh(self.transport, self.credentials, self.clock)
        self.store.save(self.credentials)

    def _send(self, method: str, url: str, body: Any) -> HttpResponse:
        if self.credentials.expired(self.clock()):
            self._refresh()
        headers = {"Authorization": f"Bearer {self.credentials.access_token}"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        self.requests += 1
        return self.transport.request(method, url, headers, data, TIMEOUT_S)

    def request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        url = path if path.startswith("https://") else API_ROOT + path
        refreshed = False
        for attempt in range(len(BACKOFF_S) + 1):
            try:
                response = self._send(method, url, body)
            except OSError as error:
                response, reason, wait = None, f"network error: {error}", None
            else:
                if 200 <= response.status < 300:
                    return json.loads(response.body) if response.body else {}
                if response.status == 401 and not refreshed:
                    self._refresh()
                    refreshed = True
                    continue
                reason, wait = self._classify(response)
            if method != "GET" and not (response and response.status == 429):
                # A lost response may hide an applied write; callers resume from remote state.
                raise SpotifyError(f"Spotify {method} did not confirm ({reason}); re-run to resume")
            if attempt == len(BACKOFF_S):
                raise (RateLimitedError if response and response.status == 429 else SpotifyError)(
                    f"Spotify request failed after {attempt} retries ({reason})"
                )
            self.sleep(wait if wait is not None else BACKOFF_S[attempt])
        raise AssertionError("unreachable")

    def _classify(self, response: HttpResponse) -> tuple[str, float | None]:
        """(reason, wait) for a retryable failure; raises for anything not worth retrying."""
        if response.status == 429:
            detail = _error(response)
            if detail.get("reason") == "QUOTA_EXCEEDED":
                raise RateLimitedError(
                    "the Spotify development-mode quota is spent; re-run later to continue"
                )
            wait = response.retry_after
            if wait is not None and wait > MAX_RETRY_AFTER_S:
                raise RateLimitedError(f"Spotify asked to wait {wait:.0f} s; re-run later")
            return "rate limited (HTTP 429)", wait
        if response.status in _RETRYABLE:
            return f"HTTP {response.status}", None
        message = _error(response).get("message", "")
        if response.status == 403:
            raise SpotifyError(
                f"Spotify refused the request (HTTP 403{': ' + message if message else ''}); "
                "development-mode apps work only for users on the app's allowlist, and the "
                "app owner needs Spotify Premium"
            )
        raise SpotifyError(f"Spotify rejected the request (HTTP {response.status}) {message}")

    def paginate(self, path: str) -> Iterator[dict[str, Any]]:
        """Every item of a paged listing, following ``next`` links."""
        url: str | None = f"{path}{'&' if '?' in path else '?'}limit={PAGE_SIZE}"
        while url:
            page = self.request("GET", url)
            yield from page.get("items") or ()
            url = page.get("next")

    def me(self) -> dict[str, Any]:
        return self.request("GET", "/me")

    def search_isrc(self, isrc: str) -> list[dict[str, Any]]:
        query = urlencode({"q": f"isrc:{isrc}", "type": "track", "limit": SEARCH_LIMIT})
        return list(self.request("GET", f"/search?{query}").get("tracks", {}).get("items") or ())

    def create_playlist(self, name: str, description: str) -> dict[str, Any]:
        body = {"name": name, "description": description, "public": False}
        return self.request("POST", "/me/playlists", body)

    def add_items(self, playlist_id: str, uris: list[str]) -> None:
        self.request("POST", f"/playlists/{playlist_id}/items", {"uris": uris})


def _error(response: HttpResponse) -> dict[str, Any]:
    try:
        detail = json.loads(response.body).get("error", {})
    except (ValueError, AttributeError):
        return {}
    return detail if isinstance(detail, dict) else {}
