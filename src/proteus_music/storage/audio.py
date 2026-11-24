"""Cache of per-file acoustic features and fingerprints, keyed by file identity and version.

A file's identity is its path, size and modification time; any change, or a new extractor
version, misses the cache. Rows for an older identity of the same path are replaced.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FileIdentity:
    path: str
    size_bytes: int
    modified_ns: int

    @classmethod
    def of(cls, path: str) -> FileIdentity | None:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return cls(path, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class AnalysisRow:
    status: str  # "ok", or "failed" when decoding or analysis was impossible.
    codec: str | None
    sample_rate: int | None
    features: tuple[float, ...] | None
    tempo_bpm: float | None
    tempo_confidence: float | None
    error: str | None


def load_analyses(
    conn: sqlite3.Connection, identities: list[FileIdentity], extractor: str
) -> dict[str, AnalysisRow]:
    """Cached rows whose identity still matches the file on disk, by path."""
    found = {}
    for identity in identities:
        row = conn.execute(
            "SELECT * FROM audio_analysis WHERE path = ? AND size_bytes = ? AND modified_ns = ?"
            " AND extractor = ?",
            (identity.path, identity.size_bytes, identity.modified_ns, extractor),
        ).fetchone()
        if row is not None:
            features = tuple(json.loads(row["features"])) if row["features"] else None
            found[identity.path] = AnalysisRow(
                row["status"], row["codec"], row["sample_rate"], features,
                row["tempo_bpm"], row["tempo_confidence"], row["error"],
            )  # fmt: skip
    return found


def save_analysis(
    conn: sqlite3.Connection, identity: FileIdentity, extractor: str, row: AnalysisRow
) -> None:
    conn.execute(
        "DELETE FROM audio_analysis WHERE path = ? AND extractor = ?", (identity.path, extractor)
    )
    conn.execute(
        "INSERT INTO audio_analysis (path, size_bytes, modified_ns, extractor, status, codec,"
        " sample_rate, features, tempo_bpm, tempo_confidence, error, analyzed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            identity.path, identity.size_bytes, identity.modified_ns, extractor, row.status,
            row.codec, row.sample_rate, json.dumps(row.features) if row.features else None,
            row.tempo_bpm, row.tempo_confidence, row.error, datetime.now(UTC).isoformat(),
        ),
    )  # fmt: skip


def load_fingerprints(
    conn: sqlite3.Connection, identities: list[FileIdentity], algorithm: str
) -> dict[str, np.ndarray | None]:
    """Cached fingerprints by path (None when fingerprinting failed for that identity)."""
    found = {}
    for identity in identities:
        row = conn.execute(
            "SELECT fingerprint FROM audio_fingerprints WHERE path = ? AND size_bytes = ?"
            " AND modified_ns = ? AND algorithm = ?",
            (identity.path, identity.size_bytes, identity.modified_ns, algorithm),
        ).fetchone()
        if row is not None:
            blob = row["fingerprint"]
            found[identity.path] = None if blob is None else np.frombuffer(blob, dtype="<u4")
    return found


def save_fingerprint(
    conn: sqlite3.Connection,
    identity: FileIdentity,
    algorithm: str,
    fingerprint: list[int] | None,
    error: str | None,
) -> None:
    conn.execute(
        "DELETE FROM audio_fingerprints WHERE path = ? AND algorithm = ?",
        (identity.path, algorithm),
    )
    blob = None if fingerprint is None else np.asarray(fingerprint, dtype="<u4").tobytes()
    conn.execute(
        "INSERT INTO audio_fingerprints (path, size_bytes, modified_ns, algorithm, fingerprint,"
        " error) VALUES (?, ?, ?, ?, ?, ?)",
        (identity.path, identity.size_bytes, identity.modified_ns, algorithm, blob, error),
    )
