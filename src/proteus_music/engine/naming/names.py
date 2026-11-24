"""Baseline playlist names and descriptions from distinctive metadata (no language model)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from proteus_music.engine.features.metadata import TrackDescriptor
from proteus_music.library.identity import normalize_text

# Genre, tag and artist strings repeat across thousands of tracks; normalize each once.
_normalized = lru_cache(maxsize=1 << 16)(normalize_text)
_ROMAN = ("II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
_MIN_SHARE = 0.25

Extractor = Callable[[TrackDescriptor], Sequence[str]]
_FIELDS: dict[str, Extractor] = {
    "genre": lambda d: d.genres,
    "tag": lambda d: d.tags,
    "artist": lambda d: d.artists,
    "decade": lambda d: (d.decade,) if d.decade else (),
}


@dataclass(frozen=True)
class PlaylistLabel:
    name: str
    description: str


@dataclass(frozen=True)
class _Term:
    display: str
    share: float
    score: float


def _display(forms: Counter[str]) -> str:
    """Most common original spelling; all-lowercase tags are title-cased for display."""
    form = forms.most_common(1)[0][0]
    return form.title() if form.islower() else form


class _Vocabulary:
    """Library-wide term frequencies, used to find what makes each playlist distinctive."""

    def __init__(self, descriptors: Sequence[TrackDescriptor]) -> None:
        self.size = max(len(descriptors), 1)
        self.counts = {f: self._count(descriptors, get) for f, get in _FIELDS.items()}
        self.display: dict[str, Counter[str]] = {}
        self._terms: dict[tuple[int, str], list[_Term]] = {}
        for get in _FIELDS.values():
            for d in descriptors:
                for value in get(d):
                    self.display.setdefault(_normalized(value), Counter())[value] += 1

    @staticmethod
    def _count(descriptors: Sequence[TrackDescriptor], get: Extractor) -> Counter[str]:
        return Counter(k for d in descriptors for k in {_normalized(v) for v in get(d)} if k)

    def terms(self, group: Sequence[TrackDescriptor], field: str) -> list[_Term]:
        """Terms of ``group`` ranked by distinctiveness; cached per group object and field."""
        key = (id(group), field)
        if key not in self._terms:
            self._terms[key] = self._rank(group, field) if group else []
        return self._terms[key]

    def _rank(self, group: Sequence[TrackDescriptor], field: str) -> list[_Term]:
        local = self._count(group, _FIELDS[field])
        terms = []
        for key, count in local.items():
            share = count / len(group)
            lift = share / (self.counts[field][key] / self.size)
            terms.append(_Term(_display(self.display[key]), share, share * math.log1p(lift)))
        return sorted(terms, key=lambda t: (-t.score, t.display))


def _candidates(vocab: _Vocabulary, group: Sequence[TrackDescriptor]) -> list[str]:
    genres = [t for t in vocab.terms(group, "genre") if t.share >= _MIN_SHARE]
    tags = [t for t in vocab.terms(group, "tag") if t.share >= _MIN_SHARE]
    artists = [t for t in vocab.terms(group, "artist") if t.share >= _MIN_SHARE]
    decades = [t for t in vocab.terms(group, "decade") if t.share >= 0.5]
    primary: list[str] = []
    for terms in (genres, tags):
        if terms:
            pair = (
                terms[:2]
                if len(terms) > 1 and terms[1].score >= 0.5 * terms[0].score
                else terms[:1]
            )
            primary.append(" & ".join(t.display for t in pair))
    if artists:
        primary.append(f"{artists[0].display} & Similar")
    if decades:
        primary.append(f"{decades[0].display} Mix")
    if not primary:
        has_data = any(d.genres or d.tags or d.artists or d.year for d in group)
        primary.append("Mixed Selection" if has_data else "Unsorted: Missing Metadata")
    base = primary[0]
    qualifiers = [t.display for t in (*tags, *decades, *artists) if t.display not in base]
    return [*primary, *(f"{base} ({q})" for q in qualifiers), *(f"{base} {r}" for r in _ROMAN)]


def _describe(vocab: _Vocabulary, group: Sequence[TrackDescriptor]) -> str:
    parts = [f"{len(group)} track{'s' if len(group) != 1 else ''}."]
    for field, label, limit in (
        ("genre", "Genres", 3),
        ("tag", "Tags", 3),
        ("artist", "Artists include", 4),
    ):
        if terms := vocab.terms(group, field)[:limit]:
            parts.append(f"{label}: {', '.join(t.display for t in terms)}.")
    if decades := vocab.terms(group, "decade"):
        parts.append(f"Mostly {decades[0].display}." if decades[0].share >= 0.5 else "Mixed eras.")
    if missing := sum(1 for d in group if not (d.genres or d.tags or d.artists or d.year)):
        parts.append(f"{missing} track(s) lack analyzable metadata.")
    return " ".join(parts)


def _name_key(name: str) -> str:
    """Collision key that treats 'Hip-Hop & Rap' and 'rap and hip hop' as the same name."""
    return " ".join(sorted(normalize_text(name).split()))


def label_playlists(
    groups: Sequence[Sequence[TrackDescriptor]], fixed_names: Mapping[int, str] | None = None
) -> list[PlaylistLabel]:
    """Names and descriptions per group. ``fixed_names`` (user categories, unclassified) are
    kept verbatim; other groups get distinct names, larger groups choosing first."""
    fixed = dict(fixed_names or {})
    vocab = _Vocabulary([d for g in groups for d in g])
    names: dict[int, str] = dict(fixed)
    used = {_name_key(n) for n in fixed.values()}
    free = [i for i in range(len(groups)) if i not in fixed]
    for index in sorted(free, key=lambda i: -len(groups[i])):
        candidates = _candidates(vocab, groups[index]) if groups[index] else ["Empty"]
        name = next(
            (c for c in candidates if _name_key(c) not in used), f"{candidates[0]} {index + 1}"
        )
        used.add(_name_key(name))
        names[index] = name
    return [PlaylistLabel(names[i], _describe(vocab, g)) for i, g in enumerate(groups)]
