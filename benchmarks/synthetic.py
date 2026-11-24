"""Synthetic libraries with known groupings, for quality and performance benchmarks."""

from __future__ import annotations

import random
from collections.abc import Sequence

from proteus_music.engine.features.metadata import TrackDescriptor


def make_library(
    sizes: Sequence[int],
    *,
    seed: int = 0,
    noise: float = 0.1,
    families: int = 0,
    missing: float = 0.0,
) -> tuple[list[TrackDescriptor], list[int]]:
    """Tracks drawn from groups of the given sizes, returning descriptors and true group ids.

    Each group has its own genres, tags, artists and era. ``noise`` replaces tags with random
    ones from the whole library; ``families`` > 0 nests groups under shared family genres
    (a deliberately ambiguous structure); ``missing`` drops all genres and tags.
    """
    rng = random.Random(seed)
    all_tags = [f"tag {g}-{t}" for g in range(len(sizes)) for t in range(4)]
    descriptors, truth = [], []
    for group, size in enumerate(sizes):
        genres = [f"genre {group}-{i}" for i in range(2)]
        tags = all_tags[group * 4 : group * 4 + 4]
        artists = [f"artist {group}-{a}" for a in range(max(2, size // 8))]
        era = 1965 + (group * 7) % 55
        for _ in range(size):
            track_genres = [
                f"genre {rng.randrange(len(sizes))}-{rng.randrange(2)}"
                if rng.random() < noise
                else g
                for g in rng.sample(genres, rng.choice((1, 2)))
            ]
            if families:
                track_genres = [f"family {group % families}", *track_genres[:1]]
            track_tags = [
                rng.choice(all_tags) if rng.random() < noise else t for t in rng.sample(tags, 2)
            ]
            if rng.random() < missing:
                track_genres, track_tags = [], []
            descriptors.append(
                TrackDescriptor(
                    track_id=str(len(descriptors)),
                    genres=tuple(track_genres),
                    tags=tuple(dict.fromkeys(track_tags)),
                    artists=(rng.choice(artists),),
                    year=era + rng.randrange(-4, 5),
                )
            )
            truth.append(group)
    return descriptors, truth


def structureless(n: int, *, seed: int = 0) -> list[TrackDescriptor]:
    """Tracks whose genres and tags are independent random draws: no grouping exists."""
    rng = random.Random(seed)
    return [
        TrackDescriptor(
            track_id=str(i),
            genres=tuple(rng.sample([f"genre {g}" for g in range(30)], 3)),
            tags=tuple(rng.sample([f"tag {t}" for t in range(60)], 4)),
            artists=(f"artist {rng.randrange(max(2, n // 3))}",),
            year=rng.randrange(1960, 2025),
        )
        for i in range(n)
    ]
