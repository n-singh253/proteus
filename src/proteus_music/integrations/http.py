"""Minimal HTTPS transport shared by network adapters, replaceable by a fake in tests."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    retry_after: float | None = None


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse: ...


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header; HTTP-date forms are ignored (callers back off)."""
    return float(value) if value and value.strip().isdigit() else None


class UrllibTransport:
    """Standard-library transport; connection failures and timeouts raise OSError."""

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HttpResponse(response.status, response.read())
        except urllib.error.HTTPError as error:
            retry_after = parse_retry_after(error.headers.get("Retry-After"))
            return HttpResponse(error.code, error.read(), retry_after)
