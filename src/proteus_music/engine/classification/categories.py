"""User-defined categories as prototypes in the track feature space.

A category name is matched against the collection's own permitted genre, tag and decade
vocabulary: the whole phrase, then its individual words. The prototype therefore encodes only
what the metadata can support; a category whose terms never occur has no evidence at all.
Future semantic or acoustic embeddings can replace ``category_vector`` without touching
assignment, which only sees prototype rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from proteus_music.engine.features.metadata import FeatureMatrix, FeatureWeights
from proteus_music.engine.optimization.assignment import InfeasibleConstraintsError
from proteus_music.library.identity import normalize_text
from proteus_music.models.organization import Category, PlaylistOrigin
from proteus_music.models.quality import CategoryReport, Evidence

# A track supports a category when its cosine compatibility reaches this value; one exact
# genre or tag match on a typical track scores well above it, a lone partial word does not.
SUPPORT_THRESHOLD = 0.15
MIN_STRONG_SUPPORT = 3
_MATCHABLE_BLOCKS = ("genre", "tags", "era")
_STOPWORDS = frozenset({"a", "an", "and", "for", "in", "music", "of", "songs", "the", "to"})


@dataclass(frozen=True)
class ResolvedCategories:
    """Candidate category playlists with their prototype rows: required categories first
    (in the user's order), then supported preferred ones, strongest first."""

    categories: tuple[Category, ...]
    origins: tuple[PlaylistOrigin, ...]
    prototypes: np.ndarray
    reports: tuple[CategoryReport, ...]

    @property
    def n_required(self) -> int:
        return sum(c.required for c in self.categories)

    def for_count(self, k: int) -> ResolvedCategories:
        """The categories used at ``k`` playlists: every required one, then as many preferred
        ones as fit. Preferences fill slots; they never raise the playlist count."""
        used = max(self.n_required, min(len(self.categories), k))
        if used == len(self.categories):
            return self
        unused = {c.name for c in self.categories[used:]}
        reports = tuple(
            r.model_copy(update={"kept": False, "note": "no free slot at the selected count"})
            if r.name in unused
            else r
            for r in self.reports
        )
        return ResolvedCategories(
            self.categories[:used], self.origins[:used], self.prototypes[:used], reports
        )


def _words(name: str) -> list[str]:
    return [w for w in normalize_text(name).split() if w not in _STOPWORDS]


def category_vector(
    name: str, features: FeatureMatrix, weights: FeatureWeights
) -> tuple[np.ndarray, tuple[str, ...]]:
    phrase, words = normalize_text(name), _words(name)
    block_weight = {"genre": weights.genre, "tags": weights.tags, "era": weights.era}
    vector = np.zeros(len(features.columns))
    matched: dict[str, None] = {}
    for j, (block, token) in enumerate(features.columns):
        if block not in _MATCHABLE_BLOCKS:
            continue
        term = token.removeprefix("word:")
        strength = 1.0 if token == phrase else (1 / len(words) if term in words else 0.0)
        if strength:
            vector[j] = max(vector[j], strength * np.sqrt(block_weight[block]))
            matched[term] = None
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector), tuple(matched)


def _is_partial(name: str, terms: tuple[str, ...]) -> bool:
    """True when the phrase itself never occurs and under half of its words do."""
    words = _words(name)
    if normalize_text(name) in terms or not words:
        return False
    return sum(w in terms for w in words) * 2 < len(words)


def _evidence(support: int, min_size: int, partial: bool) -> Evidence:
    if support == 0:
        return Evidence.NONE
    if partial or support < max(min_size, MIN_STRONG_SUPPORT):
        return Evidence.WEAK
    return Evidence.STRONG


def resolve_categories(
    categories: Sequence[Category],
    features: FeatureMatrix,
    weights: FeatureWeights,
    *,
    min_size: int,
    preferred_slots: int,
) -> ResolvedCategories:
    """Score evidence, keep every required category and the best-supported preferred ones.

    A required category with no supporting track is rejected rather than filled with
    unrelated songs; a preferred one is kept only with strong evidence and a free slot.
    """
    analyzable = features.matrix[features.core]
    scored = []
    for category in categories:
        vector, terms = category_vector(category.name, features, weights)
        compat = analyzable @ vector if analyzable.shape[0] else np.zeros(0)
        support = int(np.count_nonzero(compat >= SUPPORT_THRESHOLD))
        evidence = _evidence(support, min_size, _is_partial(category.name, terms))
        scored.append((category, vector, terms, support, evidence))

    unsupported = [c.name for c, _, _, _, e in scored if c.required and e is Evidence.NONE]
    if unsupported:
        raise InfeasibleConstraintsError(
            [
                f"required category '{name}' matches no permitted genre, tag or decade in the "
                "library; rename it, add metadata, or make it a preferred category"
                for name in unsupported
            ]
        )
    preferred = sorted(
        (s for s in scored if not s[0].required and s[4] is Evidence.STRONG), key=lambda s: -s[3]
    )
    kept_preferred = {s[0].name for s in preferred[: max(preferred_slots, 0)]}

    reports = []
    for category, _, terms, support, evidence in scored:
        keep = category.required or category.name in kept_preferred
        note = ""
        if not keep:
            note = "not enough evidence" if evidence is not Evidence.STRONG else "no free slot"
        elif evidence is Evidence.WEAK and _is_partial(category.name, terms):
            note = f"only partially matched ({', '.join(terms)}); meaning may not be captured"
        elif evidence is Evidence.WEAK:
            note = f"only {support} supporting track(s); playlist may include weaker matches"
        reports.append(
            CategoryReport(
                name=category.name,
                required=category.required,
                matched_terms=terms,
                supporting_tracks=support,
                evidence=evidence,
                kept=keep,
                note=note,
            )
        )
    ordered = [s for s in scored if s[0].required]
    ordered += [s for s in preferred if s[0].name in kept_preferred]
    matrix = np.vstack([s[1] for s in ordered]) if ordered else np.zeros((0, len(features.columns)))
    origins = tuple(
        PlaylistOrigin.REQUIRED_CATEGORY if s[0].required else PlaylistOrigin.PREFERRED_CATEGORY
        for s in ordered
    )
    return ResolvedCategories(tuple(s[0] for s in ordered), origins, matrix, tuple(reports))
