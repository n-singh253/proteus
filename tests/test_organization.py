"""Organization modes on synthetic libraries with known groupings."""

import random

import pytest
from pydantic import ValidationError

from proteus_music.engine.evaluation.validation import validate_result
from proteus_music.engine.features.metadata import TrackDescriptor
from proteus_music.engine.optimization.assignment import (
    InfeasibleConstraintsError,
    check_feasibility,
)
from proteus_music.engine.pipeline import plan_playlists
from proteus_music.models.organization import (
    AssignmentFlag,
    Category,
    OrganizationConfig,
    OrganizationMode,
    PlaylistOrigin,
)
from proteus_music.models.quality import Evidence
from proteus_music.playlists.service import new_result

GROUPS = {
    "Hip-Hop": ("rap", "energetic", "bass"),
    "Deep House": ("dance", "night", "club"),
    "Classical": ("orchestral", "calm", "strings"),
    "Indie Rock": ("guitar", "melancholic", "lofi"),
}


def library(sizes=(40, 40, 40, 40), *, seed=0, featureless=0, partial=0):
    rng = random.Random(seed)
    tracks = []
    for (genre, tags), size in zip(GROUPS.items(), sizes, strict=False):
        for i in range(size):
            tracks.append(TrackDescriptor(
                f"{genre}-{i}", (genre,), tuple(rng.sample(tags, 2)),
                (f"{genre} artist {i % 4}",), 1990 + 10 * list(GROUPS).index(genre),
            ))  # fmt: skip
    tracks += [TrackDescriptor(f"none-{i}", (), (), (), None) for i in range(featureless)]
    tracks += [TrackDescriptor(f"partial-{i}", (), (), ("Hip-Hop artist 0",), None)
               for i in range(partial)]  # fmt: skip
    return tracks


def organize(tracks, **settings):
    config = OrganizationConfig(**settings)
    plan = plan_playlists(tracks, config)
    result = new_result(plan, config, tuple(t.track_id for t in tracks))
    validate_result(result)  # Exclusive, complete and constraint-satisfying in every mode.
    return result


def genre_of(track_id: str) -> str:
    return track_id.rsplit("-", 1)[0]


CATEGORIES = (Category(name="Hip-Hop"), Category(name="Classical"))


@pytest.mark.parametrize(
    "settings",
    [
        {"playlists": 4},
        {"min_playlists": 2, "max_playlists": 8},
        {"categories": CATEGORIES},
        {"playlists": 4, "categories": CATEGORIES},
        {"min_playlists": 3, "max_playlists": 6, "min_size": 30, "max_size": 60,
         "categories": CATEGORIES},
        {"mode": OrganizationMode.AUTO},
        {"mode": OrganizationMode.AUTO, "categories": (Category(name="Calm", required=False),)},
    ],
)  # fmt: skip
def test_every_mode_yields_a_valid_deterministic_result(settings):
    tracks = library(featureless=3)
    first, second = organize(tracks, **settings), organize(tracks, **settings)
    assert first.assignments == second.assignments


def test_range_and_auto_recover_clear_structure():
    tracks = library()
    ranged = organize(tracks, min_playlists=2, max_playlists=8)
    auto = organize(tracks, mode=OrganizationMode.AUTO)

    for result in (ranged, auto):
        assert len(result.playlists) == 4
        assert result.selection is not None and result.selection.chosen == 4
        assert all(len({genre_of(t) for t in p.track_ids}) == 1 for p in result.playlists)
    assert {p.name for p in auto.playlists} == set(GROUPS)


def test_small_or_imbalanced_libraries():
    assert len(organize(library((6,)), mode=OrganizationMode.AUTO).playlists) == 1
    imbalanced = organize(library((160, 60, 25, 12)), mode=OrganizationMode.AUTO)
    assert sorted(len(p.track_ids) for p in imbalanced.playlists) == [12, 25, 60, 160]


def test_size_constraints_bound_feasible_counts():
    ranged = OrganizationConfig(min_playlists=2, max_playlists=10, min_size=30)
    assert check_feasibility(100, ranged) == (2, 3)
    impossible = OrganizationConfig(min_playlists=5, max_playlists=8, min_size=3)
    with pytest.raises(InfeasibleConstraintsError, match=r"allow 1\.\.3 playlists"):
        check_feasibility(10, impossible)


def test_categories_are_preserved_and_evidence_is_honest():
    tracks = library()
    full = organize(tracks, categories=CATEGORIES)
    assert [(p.name, p.origin) for p in full.playlists] == [
        ("Hip-Hop", PlaylistOrigin.REQUIRED_CATEGORY),
        ("Classical", PlaylistOrigin.REQUIRED_CATEGORY),
    ]

    partial = organize(tracks, playlists=4, categories=CATEGORIES)
    hip_hop, classical, *discovered = partial.playlists
    assert {genre_of(t) for t in hip_hop.track_ids} == {"Hip-Hop"}
    assert {genre_of(t) for t in classical.track_ids} == {"Classical"}
    assert {p.name for p in discovered} == {"Deep House", "Indie Rock"}

    weak = organize(tracks, categories=(Category(name="Late Night Driving"),))
    assert weak.categories[0].evidence is Evidence.WEAK  # Only the word "night" matches.
    with pytest.raises(InfeasibleConstraintsError, match="'Polka' matches no"):
        organize(tracks, categories=(Category(name="Polka"),))
    dropped = organize(tracks, playlists=3, categories=(Category(name="Polka", required=False),))
    assert not dropped.categories[0].kept


def test_missing_features_are_explicit_never_guessed():
    tracks = library(featureless=4, partial=2)
    auto = organize(tracks, mode=OrganizationMode.AUTO)
    unclassified = [p for p in auto.playlists if p.origin is PlaylistOrigin.UNCLASSIFIED]
    assert len(unclassified) == 1
    assert set(unclassified[0].track_ids) == {f"none-{i}" for i in range(4)}
    flags = {a.track_id: a.flags for a in auto.assignments}
    assert AssignmentFlag.PARTIAL_FEATURES in flags["partial-0"]

    fixed = organize(tracks, playlists=4)  # Exact count: no room for an extra playlist.
    assert all(p.origin is PlaylistOrigin.DISCOVERED for p in fixed.playlists)
    featureless = [a for a in fixed.assignments if a.track_id.startswith("none")]
    assert all(AssignmentFlag.MISSING_FEATURES in a.flags for a in featureless)


@pytest.mark.parametrize(
    "settings",
    [
        {"playlists": 5, "min_playlists": 2},
        {"min_playlists": 6, "max_playlists": 3},
        {"mode": OrganizationMode.AUTO, "playlists": 4},
        {"playlists": 1, "categories": CATEGORIES},
        {"categories": (Category(name="Rock"), Category(name="rock"))},
        {},
    ],
)
def test_contradictory_configurations_are_rejected(settings):
    with pytest.raises(ValidationError):
        OrganizationConfig(**settings)


def _preferred(*names):
    return tuple(Category(name=n, required=False) for n in names)


@pytest.mark.parametrize(
    "settings",
    [{"mode": OrganizationMode.AUTO}, {"min_playlists": 1, "max_playlists": 6}],
)
def test_preferred_categories_never_force_extra_playlists(settings):
    identical = [TrackDescriptor(str(i), ("Jazz", "Rock"), (), (), None) for i in range(24)]
    plain = organize(identical, **settings)
    preferred = organize(identical, **settings, categories=_preferred("Jazz", "Rock"))

    assert len(plain.playlists) == len(preferred.playlists) == 1
    assert [c.kept for c in preferred.categories] == [True, False]  # One slot, one preference.


def test_supported_preferences_fill_available_slots():
    tracks = library()
    auto = organize(tracks, mode=OrganizationMode.AUTO, categories=_preferred("Hip-Hop", "Polka"))
    assert len(auto.playlists) == 4
    origins = {p.name: p.origin for p in auto.playlists}
    assert origins["Hip-Hop"] is PlaylistOrigin.PREFERRED_CATEGORY
    assert not auto.categories[1].kept  # Polka has no evidence.

    exact = organize(tracks, playlists=3, categories=_preferred("Hip-Hop", "Classical"))
    assert sum(p.origin is PlaylistOrigin.PREFERRED_CATEGORY for p in exact.playlists) == 2
