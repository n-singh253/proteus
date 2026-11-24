"""Metadata-only vs audio-only vs fused organization on a synthetic audio collection.

Usage: uv run python benchmarks/acoustic.py [--per-style 30] [--seconds 20] [--encode]
       [--algorithm NAME]

Generates four synthetic musical styles (WAV, or AAC/FLAC via macOS ``afconvert`` with
--encode), imports them through CSV with deliberately imperfect metadata, analyzes the audio
cold and then warm, and compares feature sets by agreement with the generating styles.
Everything is synthetic, so this measures mechanics, not listening quality.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import graph_clustering
import numpy as np
from sklearn.metrics import adjusted_rand_score

from proteus_music.integrations.files.csv_format import read_csv
from proteus_music.library.audio import analyze_audio
from proteus_music.library.service import import_batch
from proteus_music.models.organization import FeatureSet, OrganizationConfig, OrganizationMode
from proteus_music.playlists.service import organize_library
from proteus_music.storage.db import connect
from proteus_music.storage.tracks import load_tracks

graph_clustering.register()  # Experimental: --algorithm knn-graph.

SR = 22050
STYLES = ("Ambient", "Drum and Bass", "Folk", "Techno")


def _envelope(n: int, period: int, decay: float) -> np.ndarray:
    env = np.zeros(n)
    for start in range(0, n, period):
        length = min(n - start, period)
        env[start : start + length] = np.exp(-np.arange(length) / (decay * SR))
    return env


def synthesize(style: str, seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    root = 110 * 2 ** (rng.integers(0, 12) / 12)
    if style == "Ambient":  # Slow swelling harmonic pad, no beat, dark.
        x = sum(np.sin(2 * np.pi * root * k * t + rng.random() * 6) / k**2 for k in (1, 2, 3, 4))
        return 0.25 * x * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.12) * t))
    bpm = {"Drum and Bass": 174, "Folk": 95, "Techno": 128}[style] * rng.uniform(0.97, 1.03)
    beat = int(SR * 60 / bpm)
    if style == "Drum and Bass":  # Bright noisy breakbeat with sub bass.
        hats = rng.standard_normal(n) * _envelope(n, beat // 2, 0.03)
        kick = np.sin(2 * np.pi * 50 * t) * _envelope(n, beat, 0.12)
        return 0.35 * hats + 0.5 * kick + 0.15 * np.sin(2 * np.pi * root / 2 * t)
    if style == "Folk":  # Plucked harmonic strings, mid brightness, quiet.
        pluck = _envelope(n, beat, 0.4)
        tone = sum(np.sin(2 * np.pi * root * 2 * k * t) / k for k in range(1, 7))
        return 0.12 * tone * pluck + 0.005 * rng.standard_normal(n)
    kick = np.sin(2 * np.pi * 55 * t) * _envelope(n, beat, 0.15)  # Techno: kick + saw bass.
    saw = 2 * ((root * t) % 1) - 1
    return 0.6 * kick + 0.2 * saw * _envelope(n, beat // 4, 0.08)


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x / max(1.0, np.abs(x).max()), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def build_collection(root: Path, per_style: int, seconds: float, encode: bool, seed: int):
    """Write audio and a CSV whose metadata is imperfect: wrong, missing, and audio-less rows."""
    rng = np.random.default_rng(seed)
    truth, rows = {}, []
    for s, style in enumerate(STYLES):
        for i in range(per_style):
            title = f"{style} {i:03d}"
            path = root / f"{s}_{i:03d}.wav"
            write_wav(path, synthesize(style, seconds, rng))
            if encode and shutil.which("afconvert") and i % 3:
                target = path.with_suffix(".m4a" if i % 3 == 1 else ".flac")
                fmt = ["-f", "m4af", "-d", "aac"] if i % 3 == 1 else ["-f", "flac", "-d", "flac"]
                subprocess.run(["afconvert", *fmt, str(path), str(target)], check=True)
                path.unlink()
                path = target
            roll = rng.random()
            genre = STYLES[rng.integers(0, 4)] if roll < 0.25 else style  # 25% mislabeled.
            if roll > 0.70:  # 30% without genre and tags.
                genre = ""
            has_audio = rng.random() > 0.15  # 15% have no audio file.
            rows.append([title, f"{style} Artist {i % 5}", genre, str(path) if has_audio else ""])
            truth[title] = s
    with (root / "library.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "artist", "genres", "path"])
        writer.writerows(rows)
    return truth


def evaluate(conn, truth, config) -> tuple[float, list[int], float]:
    start = time.perf_counter()
    result = organize_library(conn, config)
    elapsed = time.perf_counter() - start
    titles = {t.id: t.title for t in load_tracks(conn)}
    labels = {titles[t]: p.index for p in result.playlists for t in p.track_ids}
    names = sorted(truth)
    ari = adjusted_rand_score([truth[n] for n in names], [labels[n] for n in names])
    return ari, sorted((len(p.track_ids) for p in result.playlists), reverse=True), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-style", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--encode", action="store_true", help="Mix in AAC and FLAC (macOS).")
    parser.add_argument("--algorithm", default="kmeans-merge")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        truth = build_collection(root, args.per_style, args.seconds, args.encode, seed=7)
        conn = connect(root / "bench.db")
        import_batch(conn, read_csv(root / "library.csv", separator=";"))
        cold = analyze_audio(conn)
        warm = analyze_audio(conn)
        print(f"{len(truth)} tracks, {cold.files} permitted audio files, codecs {cold.codecs}")
        print(f"analysis cold {cold.seconds:.2f} s ({cold.analyzed} decoded), ", end="")
        print(f"warm {warm.seconds:.2f} s ({warm.cached} cached, {warm.analyzed} decoded)")
        for mode, extra in (
            ("exact 4", {"playlists": 4}),
            ("auto", {"mode": OrganizationMode.AUTO}),
        ):
            for features in FeatureSet:
                runs = [
                    evaluate(
                        conn,
                        truth,
                        OrganizationConfig(
                            features=features, seed=s, algorithm=args.algorithm, **extra
                        ),
                    )
                    for s in range(args.seeds)
                ]
                aris = " ".join(f"{r[0]:.2f}" for r in runs)
                print(f"{mode:8} {features:9} ARI {aris}  sizes {runs[0][1]}  "
                      f"organize {np.mean([r[2] for r in runs]):.2f} s")  # fmt: skip


if __name__ == "__main__":
    main()
