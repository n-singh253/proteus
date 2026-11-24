"""Metadata representation: weighted TF-IDF blocks over genres, tags, artists and era.

Each block is L2-normalized and scaled by sqrt(weight / available weight), so a track's vector
has unit norm over the blocks it actually has. Cosine similarity then equals the weighted mean of
per-block similarities, renormalized for missing blocks instead of treating them as zeros.
Tracks with no analyzable metadata get a zero row and ``available = False``. Genres and tags
are the core modalities: tracks described only by artist and era are ``available`` but not
``core``, so they are matched to playlists without shaping them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from proteus_music.library.identity import normalize_text
from proteus_music.models.track import Track

FEATURE_VERSION = "metadata-tfidf-v1"


@dataclass(frozen=True)
class FeatureWeights:
    """Experimental starting weights; acoustic and mood modalities are not yet available."""

    genre: float = 0.35
    tags: float = 0.25
    artist: float = 0.10
    era: float = 0.05


@dataclass(frozen=True)
class TrackDescriptor:
    """Provider-independent, analysis-permitted description of one canonical track."""

    track_id: str
    genres: tuple[str, ...]
    tags: tuple[str, ...]
    artists: tuple[str, ...]
    year: int | None

    @property
    def decade(self) -> str | None:
        return f"{self.year // 10 * 10}s" if self.year else None


@dataclass(frozen=True)
class FeatureMatrix:
    """Unit-norm rows (zero when unavailable); ``columns`` names each column (block, token).

    ``core`` marks rows with genres or tags; if no row has them, it equals ``available``.
    """

    track_ids: tuple[str, ...]
    matrix: sparse.csr_matrix
    available: np.ndarray
    core: np.ndarray
    columns: tuple[tuple[str, str], ...] = ()
    version: str = FEATURE_VERSION


def term_presence(features: FeatureMatrix) -> sparse.csr_matrix:
    """Binary track x term matrix of whole genre and tag phrases (no partial words)."""
    columns = [
        j
        for j, (block, token) in enumerate(features.columns)
        if block in ("genre", "tags") and not token.startswith("word:")
    ]
    return (features.matrix[:, columns] != 0).astype(np.float64).tocsr()


def describe(track: Track) -> TrackDescriptor:
    """Collect metadata only from records whose source permits analysis."""
    records = track.analysis_records
    return TrackDescriptor(
        track_id=track.id,
        genres=tuple(dict.fromkeys(g for r in records for g in r.genres)),
        tags=tuple(dict.fromkeys(t for r in records for t in r.tags)),
        artists=tuple(dict.fromkeys(a for r in records for a in r.artists)),
        year=next((r.year for r in records if r.year), None),
    )


def _phrase_tokens(values: Sequence[str]) -> list[str]:
    """Whole phrases plus their words, so 'deep house' partially matches 'house'."""
    tokens: list[str] = []
    for value in values:
        phrase = normalize_text(value)
        if phrase:
            tokens.append(phrase)
            words = phrase.split()
            if len(words) > 1:
                tokens.extend(f"word:{w}" for w in words)
    return tokens


def _identity(tokens: list[str]) -> list[str]:
    return tokens


def build_features(
    descriptors: Sequence[TrackDescriptor], weights: FeatureWeights = FeatureWeights()
) -> FeatureMatrix:
    blocks = [
        ("genre", [_phrase_tokens(d.genres) for d in descriptors], weights.genre),
        ("tags", [_phrase_tokens(d.tags) for d in descriptors], weights.tags),
        (
            "artist",
            [[normalize_text(a) for a in d.artists if normalize_text(a)] for d in descriptors],
            weights.artist,
        ),
        ("era", [[d.decade] if d.decade else [] for d in descriptors], weights.era),
    ]
    n = len(descriptors)
    parts: list[sparse.csr_matrix] = []
    columns: list[tuple[str, str]] = []
    available_weight = np.zeros(n)
    core = np.zeros(n, dtype=bool)
    for block, docs, weight in blocks:
        present = np.fromiter((bool(doc) for doc in docs), dtype=bool, count=n)
        if weight <= 0 or not present.any():
            continue
        vectorizer = TfidfVectorizer(analyzer=_identity, sublinear_tf=True, norm="l2")
        parts.append(vectorizer.fit_transform(docs) * np.sqrt(weight))
        columns.extend((block, token) for token in vectorizer.get_feature_names_out())
        available_weight += weight * present
        if block in ("genre", "tags"):
            core |= present
    available = available_weight > 0
    if not core.any():
        core = available
    if not parts:
        return FeatureMatrix(
            tuple(d.track_id for d in descriptors), sparse.csr_matrix((n, 0)), available, core
        )
    scale = np.zeros(n)
    scale[available] = 1.0 / np.sqrt(available_weight[available])
    matrix = sparse.diags(scale) @ sparse.hstack(parts, format="csr")
    ids = tuple(d.track_id for d in descriptors)
    return FeatureMatrix(ids, matrix.tocsr(), available, core, tuple(columns))
