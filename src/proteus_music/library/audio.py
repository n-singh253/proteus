"""Acoustic analysis of authorized local audio, cached by file identity and extractor version.

Only local files referenced by analysis-permitted source records are decoded. Decoding runs
natively in bounded batches (mono excerpts of at most ``excerpt_seconds``); every outcome,
including failures, is cached so unchanged files are never decoded twice.
"""

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from proteus_music import _native
from proteus_music.engine.features.acoustic import (
    EXTRACTOR_VERSION,
    AcousticFeatures,
    UnanalyzableAudio,
    extract,
)
from proteus_music.models.track import Track
from proteus_music.storage import audio as audio_store
from proteus_music.storage.audio import AnalysisRow, FileIdentity
from proteus_music.storage.tracks import load_tracks

EXCERPT_SECONDS = 30.0
BATCH_SIZE = 16


@dataclass(frozen=True)
class AnalysisSummary:
    files: int
    cached: int
    analyzed: int
    failed: int
    missing_files: int
    codecs: dict[str, int]  # Decoded files by codec.
    failures: tuple[tuple[str, str], ...]
    seconds: float


def permitted_audio(tracks: Sequence[Track]) -> dict[str, str]:
    """Track ID -> local path, only for files that a permitted source record references."""
    paths = {}
    for track in tracks:
        path = next((r.local_path for r in track.analysis_records if r.local_path), None)
        if path:
            paths[track.id] = path
    return paths


def excerpt_offset(duration_ms: int | None, excerpt_seconds: float) -> float:
    """Start of the analysis window: centred in tracks long enough, else the beginning."""
    if not duration_ms or duration_ms / 1000 <= excerpt_seconds + 20:
        return 0.0
    return (duration_ms / 1000 - excerpt_seconds) / 2


def _row(decoded: _native.DecodedAudio) -> AnalysisRow:
    samples = np.frombuffer(decoded.samples, dtype="<f4")
    try:
        f = extract(samples, decoded.sample_rate)
    except UnanalyzableAudio as error:
        return AnalysisRow(
            "failed", decoded.codec, decoded.sample_rate, None, None, None, str(error)
        )
    return AnalysisRow(
        "ok", decoded.codec, decoded.sample_rate, f.values, f.tempo_bpm, f.tempo_confidence, None
    )


def analyze_audio(
    conn: sqlite3.Connection,
    *,
    batch_size: int = BATCH_SIZE,
    excerpt_seconds: float = EXCERPT_SECONDS,
    max_threads: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> AnalysisSummary:
    start = time.perf_counter()
    tracks = {t.id: t for t in load_tracks(conn)}
    paths = permitted_audio(list(tracks.values()))
    by_path = {p: tracks[t] for t, p in paths.items()}
    identities = [i for i in map(FileIdentity.of, by_path) if i is not None]
    cached = audio_store.load_analyses(conn, identities, EXTRACTOR_VERSION)
    todo = [i for i in identities if i.path not in cached]
    rows: dict[str, AnalysisRow] = dict(cached)
    for begin in range(0, len(todo), batch_size):
        batch = todo[begin : begin + batch_size]
        offsets = [excerpt_offset(by_path[i.path].duration_ms, excerpt_seconds) for i in batch]
        decoded, failures = _native.decode_audio(
            [i.path for i in batch], offsets, excerpt_seconds, max_threads
        )
        new = {d.path: _row(d) for d in decoded}
        new |= {
            f.path: AnalysisRow("failed", None, None, None, None, None, f.reason) for f in failures
        }
        with conn:
            for identity in batch:
                audio_store.save_analysis(conn, identity, EXTRACTOR_VERSION, new[identity.path])
        rows |= new
        if progress:
            progress(min(begin + batch_size, len(todo)), len(todo))
    failed = [(p, r.error or "unknown") for p, r in rows.items() if r.status != "ok"]
    return AnalysisSummary(
        files=len(identities),
        cached=len(cached),
        analyzed=len(todo),
        failed=len(failed),
        missing_files=len(by_path) - len(identities),
        codecs=dict(Counter(r.codec for r in rows.values() if r.codec and r.status == "ok")),
        failures=tuple(sorted(failed)),
        seconds=time.perf_counter() - start,
    )


def load_acoustic(conn: sqlite3.Connection, tracks: Sequence[Track]) -> dict[str, AcousticFeatures]:
    """Cached features for tracks whose permitted file is unchanged since analysis."""
    paths = permitted_audio(tracks)
    identities = [i for i in map(FileIdentity.of, paths.values()) if i is not None]
    rows = audio_store.load_analyses(conn, identities, EXTRACTOR_VERSION)
    features = {}
    for track_id, path in paths.items():
        row = rows.get(path)
        if row is not None and row.status == "ok" and row.features:
            features[track_id] = AcousticFeatures(
                row.features, row.tempo_bpm, row.tempo_confidence or 0.0
            )
    return features
