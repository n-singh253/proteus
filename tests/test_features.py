import numpy as np
import pytest

from proteus_music.engine.features.metadata import TrackDescriptor, build_features, describe
from proteus_music.models.track import Track, TrackRecord


def test_restricted_sources_never_reach_analysis():
    restricted = TrackRecord(
        source="provider", source_key="1", title="Song", genres=("Rock",), analysis_permitted=False
    )
    track = Track(
        id="t", title="Song", artists=(), album=None, duration_ms=None, year=None, isrc=None,
        musicbrainz_recording_id=None, version=None, records=(restricted,),
    )  # fmt: skip

    descriptor = describe(track)

    assert descriptor.genres == () and descriptor.artists == ()
    assert not build_features([descriptor]).available[0]


def test_missing_blocks_are_renormalized_not_zero_filled():
    full = TrackDescriptor("a", ("house",), ("night",), ("X",), 2015)
    genre_only = TrackDescriptor("b", ("house",), (), (), None)
    empty = TrackDescriptor("c", (), (), (), None)

    features = build_features([full, genre_only, empty])

    norms = np.sqrt(np.asarray(features.matrix.multiply(features.matrix).sum(axis=1)).ravel())
    assert norms == pytest.approx([1.0, 1.0, 0.0])
    assert features.available.tolist() == [True, True, False]
