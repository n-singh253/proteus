"""Local web-app protections: loopback hosts, a per-session token, and path validation.

Binding to loopback is not enough on its own: any web page the user visits can send
requests to localhost. Every API call must therefore carry the session token, which is
embedded only in the page this server renders (other origins cannot read it), and the Host
header must name this server, which defeats DNS rebinding.
"""

from __future__ import annotations

import hmac
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

TOKEN_HEADER = "X-Proteus-Token"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class PathRejectedError(ValueError):
    """A user-supplied path failed validation; the message says how to fix it."""


def allowed_hosts(host: str, port: int) -> frozenset[str]:
    names = {host, *(LOOPBACK_HOSTS if host in LOOPBACK_HOSTS else ())}
    return frozenset(f"[{n}]:{port}" if ":" in n else f"{n}:{port}" for n in names)


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, token: str, hosts: Iterable[str]) -> None:
        super().__init__(app)
        self.token = token
        self.hosts = frozenset(hosts)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        host = request.headers.get("host", "")
        if host not in self.hosts:
            return JSONResponse({"error": f"host {host!r} is not allowed"}, status_code=400)
        origin = request.headers.get("origin")
        if origin is not None and origin.split("://", 1)[-1] not in self.hosts:
            return JSONResponse({"error": "cross-origin requests are not allowed"}, 403)
        if request.url.path.startswith("/api/"):
            supplied = request.headers.get(TOKEN_HEADER, "")
            if not hmac.compare_digest(supplied, self.token):
                return JSONResponse({"error": "missing or invalid session token"}, 403)
        return await call_next(request)


def validate_path(
    raw: str,
    *,
    kind: Literal["file", "directory", "output"],
    roots: Iterable[Path],
    suffixes: Iterable[str] = (),
) -> Path:
    """Resolve an absolute path inside an allowed root and check that it fits ``kind``."""
    if not raw.strip():
        raise PathRejectedError("enter a path")
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        raise PathRejectedError(f"use an absolute path (for example {Path.home()})")
    path = path.resolve()
    roots = [Path(r).resolve() for r in roots]
    if not any(path.is_relative_to(r) for r in roots):
        allowed = ", ".join(map(str, roots))
        raise PathRejectedError(f"{path} is outside the allowed folders ({allowed})")
    if kind == "file":
        if not path.is_file():
            raise PathRejectedError(f"no file at {path}")
        allowed_suffixes = {s.lower() for s in suffixes}
        if allowed_suffixes and path.suffix.lower() not in allowed_suffixes:
            raise PathRejectedError(f"expected a {'/'.join(sorted(allowed_suffixes))} file")
    elif kind == "directory" and not path.is_dir():
        raise PathRejectedError(f"no folder at {path}")
    elif kind == "output":
        if path.exists() and not path.is_dir():
            raise PathRejectedError(f"{path} exists and is not a folder")
        if not path.parent.is_dir():
            raise PathRejectedError(f"parent folder {path.parent} does not exist")
    return path
