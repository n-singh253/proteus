"""Fingerprint-based duplicate candidates for review. Nothing is ever merged automatically.

Chromaprint fingerprints describe the recorded audio, so a remaster or a re-encode of the same
performance can look identical while being a distinct release the identity rules keep apart.
Candidates therefore carry each track's version descriptor, and the user decides.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from proteus_music import _native
from proteus_music.library.audio import permitted_audio
from proteus_music.storage import audio as audio_store
from proteus_music.storage.audio import FileIdentity
from proteus_music.storage.tracks import load_tracks

ALGORITHM = "chromaprint-test2-120s"
FINGERPRINT_SECONDS = 120.0
MAX_SHIFT = 80  # Fingerprint items (~0.12 s each): about 10 s of alignment slack.
MIN_OVERLAP = 0.5
DURATION_RATIO = 0.75  # Compare only tracks whose fingerprint lengths are this close.
DEFAULT_THRESHOLD = 0.6


@dataclass(frozen=True)
class DuplicateCandidate:
    track_a: str
    track_b: str
    similarity: float  # 1 = identical bits, 0 = unrelated (bit error rate of one half).
    same_version: bool


def similarity(a: np.ndarray, b: np.ndarray, max_shift: int = MAX_SHIFT) -> float:
    """Best alignment score over shifts, requiring the overlap to cover half the shorter print."""
    best = 0.0
    shorter = min(len(a), len(b))
    for shift in range(-max_shift, max_shift + 1):
        x, y = (a[shift:], b) if shift >= 0 else (a, b[-shift:])
        n = min(len(x), len(y))
        if n < MIN_OVERLAP * shorter or n == 0:
            continue
        errors = np.bitwise_count(np.bitwise_xor(x[:n], y[:n])).mean() / 32
        best = max(best, 1 - 2 * float(errors))
    return max(best, 0.0)


def fingerprint_library(
    conn: sqlite3.Connection, max_threads: int | None = None
) -> dict[str, np.ndarray]:
    """Fingerprints by track ID for permitted local files, computing only uncached ones."""
    paths = permitted_audio(load_tracks(conn))
    identities = {p: i for p in set(paths.values()) if (i := FileIdentity.of(p)) is not None}
    cached = audio_store.load_fingerprints(conn, list(identities.values()), ALGORITHM)
    todo = [p for p in identities if p not in cached]
    if todo:
        prints, failures = _native.fingerprint_audio(todo, FINGERPRINT_SECONDS, max_threads)
        with conn:
            for path, fingerprint in prints:
                audio_store.save_fingerprint(conn, identities[path], ALGORITHM, fingerprint, None)
                cached[path] = np.asarray(fingerprint, dtype="<u4")
            for failure in failures:
                audio_store.save_fingerprint(
                    conn, identities[failure.path], ALGORITHM, None, failure.reason
                )
                cached[failure.path] = None
    return {t: cached[p] for t, p in paths.items() if cached.get(p) is not None}


def duplicate_candidates(
    conn: sqlite3.Connection, threshold: float = DEFAULT_THRESHOLD, max_threads: int | None = None
) -> list[DuplicateCandidate]:
    """Pairs of distinct canonical tracks whose audio fingerprints match above ``threshold``.

    Tracks are sorted by fingerprint length and each is compared only with neighbours of
    similar length, avoiding an all-pairs comparison.
    """
    prints = fingerprint_library(conn, max_threads)
    versions = {t.id: t.version for t in load_tracks(conn)}
    order = sorted(prints, key=lambda t: len(prints[t]))
    candidates = []
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            if len(prints[a]) < DURATION_RATIO * len(prints[b]):
                break
            score = similarity(prints[a], prints[b])
            if score >= threshold:
                same = (versions.get(a) or "") == (versions.get(b) or "")
                candidates.append(DuplicateCandidate(a, b, round(score, 3), same))
    return sorted(candidates, key=lambda c: -c.similarity)
