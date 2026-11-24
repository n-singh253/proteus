"""Rate-limited MusicBrainz web-service client for identifier lookups.

Follows https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting: at most one request
per second, an identifying User-Agent with contact details, and backing off on HTTP 503,
the documented throttling signal. Only ISRCs and MusicBrainz IDs are ever sent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from proteus_music import __version__
from proteus_music.integrations.http import Transport

API_ROOT = "https://musicbrainz.org/ws/2/"
RECORDING_INCLUDES = "artist-credits+isrcs+releases"
MIN_INTERVAL_S = 1.0
TIMEOUT_S = 15.0
BACKOFF_S = (2.0, 4.0, 8.0)  # One entry per retry; bounds retries at three.
_RETRYABLE = {429, 500, 502, 503, 504}


class RateLimitedError(RuntimeError):
    """MusicBrainz kept throttling after every allowed retry; stop and resume later."""


class BudgetExhaustedError(RuntimeError):
    """The per-run request budget is spent; remaining lookups resume on the next run."""


class MusicBrainzError(RuntimeError):
    """A lookup failed for a non-retryable reason (e.g. a rejected query)."""


class MusicBrainzClient:
    def __init__(
        self,
        transport: Transport,
        contact: str,
        *,
        max_requests: int,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not contact.strip():
            raise ValueError("MusicBrainz requires contact details (a URL or email) in requests")
        self.user_agent = f"proteus-music/{__version__} ( {contact.strip()} )"
        self.transport, self.max_requests = transport, max_requests
        self.sleep, self.clock = sleep, clock
        self.requests = 0
        self._last: float | None = None

    def recordings_by_isrc(self, isrc: str) -> list[dict[str, Any]] | None:
        """Recordings carrying the ISRC, or None when MusicBrainz does not know it."""
        data = self._get(f"isrc/{quote(isrc)}?inc={RECORDING_INCLUDES}&fmt=json")
        return None if data is None else list(data.get("recordings", []))

    def recording(self, mbid: str) -> dict[str, Any] | None:
        return self._get(f"recording/{quote(mbid)}?inc={RECORDING_INCLUDES}&fmt=json")

    def _wait(self) -> None:
        if self._last is not None:
            remaining = MIN_INTERVAL_S - (self.clock() - self._last)
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()

    def _get(self, path: str) -> dict[str, Any] | None:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        for attempt in range(len(BACKOFF_S) + 1):
            if self.requests >= self.max_requests:
                raise BudgetExhaustedError(f"request budget of {self.max_requests} spent")
            self._wait()
            self.requests += 1
            try:
                response = self.transport.request("GET", API_ROOT + path, headers, None, TIMEOUT_S)
            except OSError as error:
                response, reason = None, f"network error: {error}"
            else:
                if response.status == 200:
                    return json.loads(response.body)
                if response.status == 404:
                    return None
                if response.status not in _RETRYABLE:
                    raise MusicBrainzError(
                        f"MusicBrainz rejected the query (HTTP {response.status})"
                    )
                reason = f"HTTP {response.status}"
            if attempt == len(BACKOFF_S):
                if response is not None and response.status in (429, 503):
                    raise RateLimitedError(f"still throttled after {attempt} retries")
                raise MusicBrainzError(f"gave up after {attempt} retries ({reason})")
            delay = BACKOFF_S[attempt]
            if response is not None and response.retry_after is not None:
                delay = min(max(delay, response.retry_after), 60.0)
            self.sleep(delay)
        raise AssertionError("unreachable")
