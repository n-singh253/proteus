"""`proteus analyze ...`: local, CPU-only analysis of authorized audio files."""

from __future__ import annotations

from typing import Annotated

import typer

from proteus_music.cli.common import MAX_LISTED_ISSUES, console, database, handled_errors
from proteus_music.library import audio
from proteus_music.library.duplicates import DEFAULT_THRESHOLD, duplicate_candidates
from proteus_music.storage.tracks import load_tracks

app = typer.Typer(help="Analyze authorized local audio (never modified).", no_args_is_help=True)


@app.command("audio")
def analyze_audio(
    ctx: typer.Context,
    batch_size: Annotated[int, typer.Option(min=1, help="Files decoded per batch.")] = 16,
    excerpt_seconds: Annotated[
        float, typer.Option(min=5, max=120, help="Seconds analyzed per file.")
    ] = audio.EXCERPT_SECONDS,
    threads: Annotated[int | None, typer.Option(min=1, help="Maximum decoder threads.")] = None,
) -> None:
    """Extract loudness, spectral, timbre and tempo features from permitted local files."""
    with handled_errors():

        def progress(done: int, total: int) -> None:
            console.print(f"  analyzed {done}/{total}", highlight=False)

        summary = audio.analyze_audio(
            database(ctx),
            batch_size=batch_size,
            excerpt_seconds=excerpt_seconds,
            max_threads=threads,
            progress=progress,
        )
        codecs = ", ".join(f"{c}: {n}" for c, n in sorted(summary.codecs.items())) or "none"
        console.print(
            f"{summary.files} permitted file(s): {summary.cached} cached, {summary.analyzed} "
            f"analyzed, {summary.failed} unanalyzable, {summary.missing_files} missing on disk "
            f"({summary.seconds:.1f} s). Decoded codecs: {codecs}.",
            highlight=False,
        )
        for path, reason in summary.failures[:MAX_LISTED_ISSUES]:
            console.print(f"  [yellow]{path}[/]: {reason}", highlight=False)


@app.command("duplicates")
def duplicates(
    ctx: typer.Context,
    threshold: Annotated[
        float, typer.Option(min=0, max=1, help="Fingerprint similarity to report (0..1).")
    ] = DEFAULT_THRESHOLD,
    threads: Annotated[int | None, typer.Option(min=1, help="Maximum decoder threads.")] = None,
) -> None:
    """List fingerprint duplicate candidates for review. Never merges tracks."""
    with handled_errors():
        conn = database(ctx)
        candidates = duplicate_candidates(conn, threshold, threads)
        titles = {t.id: f"{', '.join(t.artists) or '?'} — {t.title}" for t in load_tracks(conn)}
        for c in candidates:
            note = "" if c.same_version else " [different version descriptors: likely distinct]"
            console.print(
                f"{c.similarity:.2f}  {titles[c.track_a]}  ⇄  {titles[c.track_b]}{note}",
                highlight=False,
            )
        console.print(f"{len(candidates)} candidate pair(s). No tracks were merged.")
