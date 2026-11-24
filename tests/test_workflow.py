"""End-to-end CLI workflow: import (including the native scanner), organize, validate, export."""

import csv
import io
import json
import struct
import wave
from pathlib import Path

from typer.testing import CliRunner

from proteus_music.cli.main import app
from proteus_music.engine.evaluation.validation import validate_result
from proteus_music.engine.features.metadata import describe
from proteus_music.playlists.service import get_result
from proteus_music.storage.db import connect
from proteus_music.storage.tracks import load_tracks

runner = CliRunner()
GENRES = {"Deep House": "dance;night", "Hip-Hop": "rap;energetic", "Classical": "orchestral;calm"}


def write_tagged_wav(path: Path, title: str, artist: str, genre: str) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000)
    data = bytearray(buffer.getvalue())
    info = b"INFO"
    for chunk_id, text in ((b"INAM", title), (b"IART", artist), (b"IGNR", genre)):
        value = text.encode() + b"\x00"
        value += b"\x00" * (len(value) % 2)
        info += chunk_id + struct.pack("<I", len(value)) + value
    data += b"LIST" + struct.pack("<I", len(info)) + info
    struct.pack_into("<I", data, 4, len(data) - 8)
    path.write_bytes(bytes(data))


def write_library_csv(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Track Name", "Artist Name(s)", "Duration (ms)", "Genres", "Tags"])
        for genre, tags in GENRES.items():
            for i in range(6):
                writer.writerow(
                    [f"{genre} {i}", f"{genre} Artist {i % 2}", 200_000 + i, genre, tags]
                )
        writer.writerow(["", "Nobody", "", "", ""])  # Invalid: no title.


def invoke(db: Path, *args: str):
    return runner.invoke(app, ["--db", str(db), *args])


def test_import_organize_export_round_trip(tmp_path: Path):
    db, music, out = tmp_path / "lib.db", tmp_path / "music", tmp_path / "out"
    (music / "nested").mkdir(parents=True)
    write_tagged_wav(music / "a.wav", "Drift", "Tide Lab", "Ambient")
    write_tagged_wav(music / "nested" / "misnamed.mp3", "Bloom", "Tide Lab", "Ambient")
    (music / "broken.flac").write_bytes(b"not audio")
    write_library_csv(tmp_path / "library.csv")

    imported = invoke(db, "import", "csv", str(tmp_path / "library.csv"))
    assert imported.exit_code == 0 and "18 new track(s)" in imported.output
    assert "1 skipped" in imported.output
    scanned = invoke(db, "import", "local", str(music))
    assert scanned.exit_code == 0 and "2 new track(s)" in scanned.output
    assert "broken.flac" in scanned.output

    assert invoke(db, "organize", "-n", "4", "--min-size", "2", "--max-size", "7").exit_code == 0
    result = get_result(connect(db))
    validate_result(result)
    assert len(result.snapshot_track_ids) == 20
    assert {p.name for p in result.playlists} == {"Deep House", "Hip-Hop", "Classical", "Ambient"}

    partial = invoke(db, "export", "--target", "m3u8", "--out", str(out))
    assert partial.exit_code == 3 and "18 track(s) could not be exported" in partial.output
    m3u_text = next(out.glob("*Ambient.m3u8")).read_text()
    assert str((music / "nested" / "misnamed.mp3").resolve()) in m3u_text
    assert invoke(db, "export", "--target", "json", "--out", str(out)).exit_code == 0
    assert (
        invoke(db, "export", "--target", "json", "--out", str(out)).exit_code == 1
    )  # No overwrite.

    exported = json.loads((out / "organization.json").read_text())
    assert sum(len(p["tracks"]) for p in exported["playlists"]) == 20
    copy_db = tmp_path / "copy.db"
    assert (
        "20 new track(s)"
        in invoke(copy_db, "import", "json", str(out / "organization.json")).output
    )

    auto = invoke(db, "organize", "--mode", "auto")
    assert auto.exit_code == 0 and "Constraints satisfied" in auto.output


def test_identification_only_survives_export_and_reimport(tmp_path: Path):
    db, out = tmp_path / "lib.db", tmp_path / "out"
    write_library_csv(tmp_path / "library.csv")
    invoke(db, "import", "csv", str(tmp_path / "library.csv"), "--identification-only")
    invoke(db, "organize", "-n", "1")
    for target in ("json", "csv"):
        assert invoke(db, "export", "--target", target, "--out", str(out)).exit_code == 0

    for name, exported in (("json", "organization.json"), ("csv", "playlists.csv")):
        copy = tmp_path / f"{name}.db"
        invoke(copy, "import", name, str(out / exported))
        assert "identification-only (excluded from analysis): 18" in invoke(copy, "library").output


def test_infeasible_request_exits_with_reasons(tmp_path: Path):
    write_library_csv(tmp_path / "library.csv")
    db = tmp_path / "lib.db"
    invoke(db, "import", "csv", str(tmp_path / "library.csv"))

    result = invoke(db, "organize", "-n", "50")

    assert result.exit_code == 2
    assert "need 50 tracks" in result.output


def test_refresh_never_lifts_a_restriction(tmp_path: Path):
    db = tmp_path / "lib.db"
    restricted = tmp_path / "provider.json"
    restricted.write_text(
        json.dumps([{"id": "p1", "title": "Song", "artists": ["A"], "genres": ["Jazz"]}])
    )
    own = tmp_path / "own.csv"
    own.write_text("title,artist,genres\nSong,A,Rock\n")

    invoke(db, "import", "json", str(restricted), "--identification-only")
    assert "1 refreshed" in invoke(db, "import", "json", str(restricted)).output
    assert "identification-only (excluded from analysis): 1" in invoke(db, "library").output

    invoke(db, "import", "csv", str(own))  # Independently permitted record of the same track.
    (track,) = load_tracks(connect(db))
    assert len(track.records) == 2
    assert describe(track).genres == ("Rock",)  # Only the permitted record contributes.
