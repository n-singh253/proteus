"""Smoke-test an installed Proteus distribution: import, analyze, organize, report, export, serve.

Usage: python tools/release_smoke.py   (run with the interpreter of a clean environment where
the wheel is installed; it must not be able to import Proteus from a source checkout)

Everything runs in a temporary directory with synthetic audio and a temporary database; no
network access is used. Exits non-zero with the failing step on any failure.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

import proteus_music

PORT = 8799
STYLES = {"Drone": (110, 0.0), "Pulse": (220, 2.0), "Chirp": (440, 6.0)}


def write_audio(folder: Path) -> list[tuple[str, str, Path]]:
    """Six short WAV files per style: (title, genre, path)."""
    rng, rows = np.random.default_rng(0), []
    t = np.arange(22050 * 6) / 22050
    for genre, (pitch, beat) in STYLES.items():
        for i in range(6):
            envelope = 1.0 if not beat else (np.sin(np.pi * beat * t) > 0).astype(float)
            x = 0.4 * np.sin(2 * np.pi * pitch * (1 + i / 50) * t) * envelope
            x += 0.01 * rng.standard_normal(len(t))
            path = folder / f"{genre}_{i}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(22050)
                w.writeframes((x * 32767).astype("<i2").tobytes())
            rows.append((f"{genre} {i}", genre, path))
    return rows


def run(proteus: str, *args: str, allowed: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run([proteus, *args], capture_output=True, text=True)
    if result.returncode not in allowed:
        sys.exit(f"`proteus {' '.join(args)}` failed ({result.returncode}):\n{result.stderr}")
    print(f"ok: proteus {' '.join(args)}")
    return result.stdout


def check_server(proteus: str) -> None:
    server = subprocess.Popen([proteus, "serve", "--port", str(PORT), "--no-browser"])
    try:
        page = ""
        for _ in range(60):
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/").read().decode()
                break
            except OSError:
                time.sleep(0.5)
        token = re.search(r'name="proteus-token" content="([^"]+)"', page)
        if not token or token.group(1) == "__PROTEUS_TOKEN__" or "/assets/" not in page:
            sys.exit("serve: the bundled web interface was not served with a session token")
        library = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/library")
        try:
            urllib.request.urlopen(library)
            sys.exit("serve: an API request without the session token was accepted")
        except urllib.error.HTTPError as error:
            if error.code != 403:
                sys.exit(f"serve: expected 403 without a token, got {error.code}")
        library.add_header("X-Proteus-Token", token.group(1))
        stats = json.loads(urllib.request.urlopen(library).read())
        print(f"ok: proteus serve (web bundle served, token enforced, {stats['tracks']} tracks)")
    finally:
        server.terminate()
        server.wait(timeout=30)


def main() -> None:
    package = Path(proteus_music.__file__).resolve()
    print(f"Proteus {proteus_music.__version__} from {package.parent}")
    if "site-packages" not in package.parts:
        sys.exit("refusing to test a source checkout; install the wheel into a clean environment")
    proteus = shutil.which("proteus", path=str(Path(sys.executable).parent))
    if proteus is None:
        sys.exit("the `proteus` command was not installed next to this interpreter")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        os.environ["PROTEUS_DB"] = str(root / "proteus.db")
        (root / "audio").mkdir()
        rows = write_audio(root / "audio")
        with (root / "library.csv").open("w", newline="") as f:  # A separate portable list.
            writer = csv.writer(f)
            writer.writerow(["title", "artist", "genres"])
            writer.writerows([f"Listed {title}", "Synthetic", genre] for title, genre, _ in rows)
        run(proteus, "import", "local", str(root / "audio"))
        run(proteus, "import", "csv", str(root / "library.csv"))
        print(run(proteus, "analyze", "audio").strip())
        result = json.loads(run(proteus, "organize", "--mode", "auto", "--json"))
        assigned = sum(len(p["track_ids"]) for p in result["playlists"])
        if assigned != len(result["snapshot_track_ids"]):
            sys.exit("organize: not every track was assigned exactly once")
        run(proteus, "report", "--tracks")
        for target in ("json", "csv"):
            run(proteus, "export", "--target", target, "--out", str(root / "out" / target))
        # Listed tracks have no files, so M3U is partial (exit 3) and must say so.
        m3u = run(proteus, "export", "--target", "m3u8", "--out", str(root / "out"), allowed=(3,))
        if "18 track(s) could not be exported" not in m3u:
            sys.exit(f"export m3u8: unexportable tracks were not reported:\n{m3u}")
        print(f"organized {assigned} tracks into {len(result['playlists'])} playlists")
        check_server(proteus)
    print("release smoke test passed")


if __name__ == "__main__":
    main()
