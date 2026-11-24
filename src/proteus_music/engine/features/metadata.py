"""Metadata representation: weighted TF-IDF blocks over genres, tags, artists and era.

Each block is L2-normalized and scaled by sqrt(weight / available weight), so a track's vector
has unit norm over the blocks it actually has. Cosine similarity then equals the weighted mean of
per-block similarities, renormalized for missing blocks instead of treating them as zeros.
Tracks with no analyzable metadata get a zero row and ``available = False``. Genres, tags and
acoustic features are the core modalities: tracks described only by artist, era or tempo are
``available`` but not ``core``, so they are matched to playlists without shaping them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from proteus_music.engine.features.acoustic import (
    EXTRACTOR_VERSION,
    FEATURE_NAMES,
    AcousticFeatures,
    acoustic_block,
    tempo_bin_names,
    tempo_block,
)
from proteus_music.library.identity import normalize_text
from proteus_music.models.organization import FeatureSet
from proteus_music.models.track import Track

FEATURE_VERSION = "metadata-tfidf-v1"


@dataclass(frozen=True)
class FeatureWeights:
    """Experimental starting weights per block (tags stand in for mood). Blocks a track lacks
    are renormalized away, so these are relative, not absolute, contributions."""

    genre: float = 0.35
    tags: float = 0.25
    artist: float = 0.10
    era: float = 0.05
    acoustic: float = 0.30
    tempo: float = 0.05

    @classmethod
    def for_features(cls, features: FeatureSet) -> FeatureWeights:
        if features is FeatureSet.METADATA:
            return cls(acoustic=0.0, tempo=0.0)
        if features is FeatureSet.AUDIO:
            return cls(genre=0.0, tags=0.0, artist=0.0, era=0.0)
        return cls()


@dataclass(frozen=True)
class TrackDescriptor:
    """Provider-independent, analysis-permitted description of one canonical track."""

    track_id: str
    genres: tuple[str, ...]
    tags: tuple[str, ...]
    artists: tuple[str, ...]
    year: int | None
    acoustic: AcousticFeatures | None = None  # Only from authorized local audio.

    @property
    def decade(self) -> str | None:
        return f"{self.year // 10 * 10}s" if self.year else None


@dataclass(frozen=True)
class FeatureMatrix:
    """Unit-norm rows (zero when unavailable); ``columns`` names each column (block, token).

    ``core`` marks rows with genres, tags or acoustic features; if no row has any, it equals
    ``available``.
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


def describe(track: Track, acoustic: AcousticFeatures | None = None) -> TrackDescriptor:
    """Collect metadata only from records whose source permits analysis. ``acoustic`` must
    come from a file referenced by a permitted record (see ``library.audio``)."""
    records = track.analysis_records
    return TrackDescriptor(
        track_id=track.id,
        genres=tuple(dict.fromkeys(g for r in records for g in r.genres)),
        tags=tuple(dict.fromkeys(t for r in records for t in r.tags)),
        artists=tuple(dict.fromkeys(a for r in records for a in r.artists)),
        year=next((r.year for r in records if r.year), None),
        acoustic=acoustic,
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


def _dense_blocks(descriptors: Sequence[TrackDescriptor], weights: FeatureWeights):
    """(block, matrix, column names, present mask, weight) for the acoustic modalities."""
    acoustic = [d.acoustic for d in descriptors]
    candidates = [
        ("acoustic", acoustic_block([a.values if a else None for a in acoustic]),
         list(FEATURE_NAMES), weights.acoustic),
        ("tempo", tempo_block([a.tempo_bpm if a else None for a in acoustic]),
         tempo_bin_names(), weights.tempo),
    ]  # fmt: skip
    for block, values, names, weight in candidates:
        if values is not None and weight > 0:
            present = np.linalg.norm(values, axis=1) > 0
            yield block, sparse.csr_matrix(values), names, present, weight


def _sparse_blocks(descriptors: Sequence[TrackDescriptor], weights: FeatureWeights):
    docs_by_block = [
        ("genre", [_phrase_tokens(d.genres) for d in descriptors], weights.genre),
        ("tags", [_phrase_tokens(d.tags) for d in descriptors], weights.tags),
        (
            "artist",
            [[normalize_text(a) for a in d.artists if normalize_text(a)] for d in descriptors],
            weights.artist,
        ),
        ("era", [[d.decade] if d.decade else [] for d in descriptors], weights.era),
    ]
    for block, docs, weight in docs_by_block:
        present = np.fromiter((bool(doc) for doc in docs), dtype=bool, count=len(docs))
        if weight <= 0 or not present.any():
            continue
        vectorizer = TfidfVectorizer(analyzer=_identity, sublinear_tf=True, norm="l2")
        matrix = vectorizer.fit_transform(docs)
        yield block, matrix, list(vectorizer.get_feature_names_out()), present, weight


def build_features(
    descriptors: Sequence[TrackDescriptor], weights: FeatureWeights = FeatureWeights()
) -> FeatureMatrix:
    n = len(descriptors)
    parts: list[sparse.csr_matrix] = []
    columns: list[tuple[str, str]] = []
    available_weight = np.zeros(n)
    core = np.zeros(n, dtype=bool)
    versions: dict[str, None] = {}
    blocks = (*_sparse_blocks(descriptors, weights), *_dense_blocks(descriptors, weights))
    for block, matrix, names, present, weight in blocks:
        parts.append(matrix * np.sqrt(weight))
        columns.extend((block, name) for name in names)
        available_weight += weight * present
        if block in ("genre", "tags", "acoustic"):
            core |= present
        versions[EXTRACTOR_VERSION if block in ("acoustic", "tempo") else FEATURE_VERSION] = None
    available = available_weight > 0
    if not core.any():
        core = available
    ids = tuple(d.track_id for d in descriptors)
    version = "+".join(versions) or FEATURE_VERSION
    if not parts:
        return FeatureMatrix(ids, sparse.csr_matrix((n, 0)), available, core, (), version)
    scale = np.zeros(n)
    scale[available] = 1.0 / np.sqrt(available_weight[available])
    matrix = sparse.diags(scale) @ sparse.hstack(parts, format="csr")
    return FeatureMatrix(ids, matrix.tocsr(), available, core, tuple(columns), version)
