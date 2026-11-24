"""Acoustic features: extraction, native decoding, caching, permissions and fusion masks."""

import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from proteus_music.engine.features.acoustic import AcousticFeatures, UnanalyzableAudio, extract
from proteus_music.engine.features.metadata import FeatureWeights, TrackDescriptor, build_features
from proteus_music.engine.pipeline import plan_playlists
from proteus_music.integrations.files.csv_format import read_csv
from proteus_music.library.audio import analyze_audio, load_acoustic
from proteus_music.library.service import import_batch
from proteus_music.models.organization import FeatureSet, OrganizationConfig
from proteus_music.storage.db import connect
from proteus_music.storage.tracks import load_tracks

SR = 22050


def clicks(bpm: float, seconds: float = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(int(SR * seconds))
    for start in np.arange(0, seconds - 0.1, 60 / bpm):
        i = int(start * SR)
        x[i : i + 400] += np.hanning(400) * rng.standard_normal(400) * 0.8
    return x


def tone(seconds: float = 12, freq: float = 220) -> np.ndarray:
    return 0.3 * np.sin(2 * np.pi * freq * np.arange(int(SR * seconds)) / SR)


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def test_tempo_is_measured_or_missing_never_guessed():
    assert extract(clicks(128), SR).tempo_bpm == pytest.approx(128, rel=0.03)
    assert extract(tone(), SR).tempo_bpm is None
    with pytest.raises(UnanalyzableAudio, match="silent"):
        extract(np.zeros(SR * 5), SR)


def test_analysis_is_cached_by_file_identity_and_respects_permissions(tmp_path):
    for name, signal in (("beat", clicks(120)), ("pad", tone()), ("restricted", clicks(90))):
        write_wav(tmp_path / f"{name}.wav", signal)
    (tmp_path / "broken.flac").write_bytes(b"not audio")
    with (tmp_path / "library.csv").open("w", newline="") as f:
        rows = [
            ["Beat", "beat.wav", "true"],
            ["Pad", "pad.wav", "true"],
            ["Broken", "broken.flac", "true"],
            ["Restricted", "restricted.wav", "false"],
        ]
        csv.writer(f).writerows([["title", "path", "analysis_permitted"], *rows])
    conn = connect(tmp_path / "lib.db")
    import_batch(conn, read_csv(tmp_path / "library.csv"))

    cold = analyze_audio(conn)
    assert (cold.files, cold.analyzed, cold.failed) == (3, 3, 1)  # Restricted file never decoded.
    assert cold.codecs == {"pcm_s16le": 2}
    warm = analyze_audio(conn)
    assert (warm.cached, warm.analyzed) == (3, 0)  # Failures are cached too.

    write_wav(tmp_path / "pad.wav", tone(freq=330))  # A changed file is a new identity.
    assert analyze_audio(conn).analyzed == 1
    features = load_acoustic(conn, load_tracks(conn))
    assert len(features) == 2 and all(f.values for f in features.values())


def test_missing_audio_contributes_no_acoustic_evidence():
    beat = AcousticFeatures(extract(clicks(120), SR).values, 120.0, 0.9)
    pad = AcousticFeatures(extract(tone(), SR).values, None, 0.0)
    descriptors = [
        TrackDescriptor("a", ("House",), (), (), None, beat),
        TrackDescriptor("b", ("House",), (), (), None, pad),
        TrackDescriptor("c", ("House",), (), (), None, None),
    ]
    fused = build_features(descriptors)
    acoustic = [j for j, (block, _) in enumerate(fused.columns) if block in ("acoustic", "tempo")]
    assert not fused.matrix[2, acoustic].nnz  # No audio: zero acoustic evidence ...
    assert np.linalg.norm(fused.matrix[2].toarray()) == pytest.approx(1.0)  # ... renormalized.
    metadata = build_features(descriptors, FeatureWeights.for_features(FeatureSet.METADATA))
    assert all(block == "genre" for block, _ in metadata.columns)


def test_audio_alone_separates_distinct_sounds():
    descriptors = []
    for i in range(6):
        for kind, signal in (("beat", clicks(170, seed=i)), ("pad", tone(freq=110 + 20 * i))):
            f = extract(signal, SR)
            acoustic = AcousticFeatures(f.values, f.tempo_bpm, f.tempo_confidence)
            descriptors.append(TrackDescriptor(f"{kind}{i}", (), (), (), None, acoustic))
    plan = plan_playlists(descriptors, OrganizationConfig(playlists=2, features=FeatureSet.AUDIO))
    kinds = sorted(sorted({t[:3] for t in p.track_ids}) for p in plan.playlists)
    assert kinds == [["bea"], ["pad"]]


def chords(roots, seconds_each=2.0, gain=1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds_each)) / SR
    return gain * np.concatenate(
        [
            sum(0.2 * np.sin(2 * np.pi * 220 * 2 ** ((r + i) / 12) * t) for i in (0, 4, 7))
            for r in roots
        ]
    )


def test_fingerprints_flag_candidates_but_never_merge(tmp_path):
    from proteus_music.library.duplicates import duplicate_candidates

    song = [0, 9, 5, 7] * 4
    signals = {
        "Song": chords(song),
        "Song (quiet copy)": chords(song, gain=0.5),  # Same recording, different level.
        "Song - Remastered": np.tanh(2.0 * chords(song)),  # Same audio, distinct release.
        "Another Song": chords([2, 4, 11, 4] * 4),
    }
    rows = [["title", "artist", "path"]]
    for i, (title, x) in enumerate(signals.items()):
        write_wav(tmp_path / f"{i}.wav", x)
        rows.append([title, "Artist", f"{i}.wav"])
    with (tmp_path / "library.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    conn = connect(tmp_path / "lib.db")
    import_batch(conn, read_csv(tmp_path / "library.csv"))

    titles = {t.id: t.title for t in load_tracks(conn)}
    found = {
        frozenset((titles[c.track_a], titles[c.track_b])): c.same_version
        for c in duplicate_candidates(conn)
    }
    assert found[frozenset(("Song", "Song (quiet copy)"))] is True
    assert found[frozenset(("Song", "Song - Remastered"))] is False  # Flagged, version differs.
    assert not any("Another Song" in pair for pair in found)
    assert len(load_tracks(conn)) == 4  # Candidates only: nothing merged.
