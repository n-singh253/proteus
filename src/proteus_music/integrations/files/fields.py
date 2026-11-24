"""Parsing and output helpers shared by portable file formats."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

_GENRE_SEPARATORS = re.compile(r"[;/|,]")
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def split_list(value: str | None, separator: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(separator) if part.strip())


def split_genres(values: Iterable[str]) -> tuple[str, ...]:
    """Split multi-genre tag values such as ``"Rock/Pop; Indie"``."""
    return tuple(g.strip() for v in values for g in _GENRE_SEPARATORS.split(v) if g.strip())


def parse_duration_ms(value: str | None, *, milliseconds: bool) -> int | None:
    """Parse ``m:ss``/``h:mm:ss`` or a number of seconds (or milliseconds when flagged)."""
    if value is None or not value.strip():
        return None
    value = value.strip()
    if ":" in value:
        seconds = 0.0
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
        return round(seconds * 1000)
    number = float(value)
    return round(number if milliseconds else number * 1000)


def parse_year(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.match(r"\s*(\d{4})", value)
    return int(match.group(1)) if match else None


def content_key(*parts: object) -> str:
    """Stable key for records without a source-provided identifier."""
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:32]


def describe_error(exc: ValidationError | ValueError) -> str:
    """One-line, user-facing reason for a rejected input record."""
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'record'}: {e['msg']}" for e in exc.errors()
        )
    return str(exc)


def resolve_local_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def safe_filename(name: str) -> str:
    cleaned = _UNSAFE_FILENAME.sub("_", name).strip(" .")
    return cleaned[:120] or "playlist"


def ensure_writable(paths: Iterable[Path], overwrite: bool) -> None:
    """Refuse to replace existing files unless the caller explicitly allows it."""
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing files (enable overwrite to replace): "
            + ", ".join(existing)
        )
