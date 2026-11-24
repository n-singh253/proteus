"""Can fingerprints separate same-recording copies from distinct versions?

Usage: uv run python benchmarks/fingerprints.py

Synthesizes a song and variants: re-encodes and level/rate changes of the same recording, and
distinct versions (radio edit, extended mix, remaster, live performance), plus an unrelated
song. Reports each variant's fingerprint similarity to the original, then runs the library
duplicate finder to show which pairs it would put up for review.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, resample_poly

from proteus_music import _native
from proteus_music.integrations.files.csv_format import read_csv
from proteus_music.library.duplicates import duplicate_candidates, similarity
from proteus_music.library.service import import_batch
from proteus_music.storage.db import connect
from proteus_music.storage.tracks import load_tracks

SR = 22050
SONG = [0, 9, 5, 7] * 8  # Scale degrees of the chord roots (I vi IV V).
OTHER = [2, 4, 11, 4] * 8


def song(progression, chord_s=2.0, timbre="sine", jitter=0.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for i, degree in enumerate(progression):
        n = int(chord_s * SR * (1 + rng.normal(0, jitter)))
        t = np.arange(n) / SR
        root = 220 * 2 ** (degree / 12)
        notes = (root, root * 2 ** (4 / 12), root * 2 ** (7 / 12))
        if timbre == "sine":
            chord = sum(np.sin(2 * np.pi * f * t) for f in notes)
        else:
            chord = sum(2 * ((f * t) % 1) - 1 for f in notes) * 0.5
        melody = np.sin(2 * np.pi * root * 2 ** ((i % 3) * 2 / 12) * 2 * t) * np.exp(-t * 2)
        kick = np.sin(2 * np.pi * 55 * t) * np.exp(-(t % 0.5) * 12)
        parts.append(0.2 * chord + 0.2 * melody + 0.3 * kick)
    return np.concatenate(parts)


def variants() -> dict[str, tuple[str, np.ndarray, int]]:
    """name -> (title, samples, sample rate)."""
    original = song(SONG)
    b, a = butter(2, 2000 / (SR / 2), "high")
    remaster = np.tanh(2.5 * (original + 0.8 * lfilter(b, a, original))) * 0.9
    live = song(SONG, chord_s=2.1, timbre="saw", jitter=0.03, seed=3)
    live = live + 0.05 * np.random.default_rng(9).standard_normal(len(live))
    cut = int(20 * SR), int(35 * SR)
    edit = np.concatenate([original[: cut[0]], original[cut[1] :]])
    extended = np.concatenate(
        [original[: 28 * SR], original[16 * SR : 28 * SR], original[28 * SR :]]
    )
    return {
        "original": ("Song", original, SR),
        "gain -6 dB": ("Song (quiet copy)", original * 0.5, SR),
        "resampled 48 kHz": ("Song (48k copy)", resample_poly(original, 320, 147), 48000),
        "radio edit (15 s cut)": ("Song (Radio Edit)", edit, SR),
        "extended mix (+12 s)": ("Song (Extended Mix)", extended, SR),
        "remaster (EQ+compression)": ("Song - 2011 Remaster", remaster, SR),
        "live performance": ("Song (Live)", live, SR),
        "different song": ("Another Song", song(OTHER, seed=1), SR),
    }


def write(path: Path, x: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            (np.clip(x / max(1.0, np.abs(x).max()), -1, 1) * 32767).astype("<i2").tobytes()
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = {}
        for i, (name, (title, x, rate)) in enumerate(variants().items()):
            files[name] = (title, root / f"{i}.wav")
            write(files[name][1], x, rate)
        if shutil.which("afconvert"):
            for name, fmt, ext in (
                ("AAC re-encode", "aac", "m4a"),
                ("FLAC re-encode", "flac", "flac"),
            ):
                target = root / f"copy.{ext}"
                container = "m4af" if ext == "m4a" else "flac"
                subprocess.run(
                    [
                        "afconvert",
                        "-f",
                        container,
                        "-d",
                        fmt,
                        str(files["original"][1]),
                        str(target),
                    ],
                    check=True,
                )
                files[name] = (f"Song ({ext.upper()} copy)", target)
        paths = [str(p) for _, p in files.values()]
        prints = dict(_native.fingerprint_audio(paths)[0])
        reference = np.asarray(prints[str(files["original"][1])], dtype="<u4")
        print(f"{'variant':28} similarity to original")
        for name, (_, path) in files.items():
            score = similarity(reference, np.asarray(prints[str(path)], dtype="<u4"))
            print(f"{name:28} {score:.3f}")

        with (root / "library.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["title", "artist", "path"])
            writer.writerows([title, "Example Artist", str(path)] for title, path in files.values())
        conn = connect(root / "lib.db")
        import_batch(conn, read_csv(root / "library.csv"))
        titles = {t.id: t.title for t in load_tracks(conn)}
        print("\nduplicate candidates (threshold 0.6); nothing is merged:")
        for c in duplicate_candidates(conn):
            verdict = "same version" if c.same_version else "different version descriptors"
            print(
                f"  {c.similarity:.3f}  {titles[c.track_a]}  <->  {titles[c.track_b]}  ({verdict})"
            )


if __name__ == "__main__":
    main()
