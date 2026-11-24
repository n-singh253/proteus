"""Human-readable rendering of results for the terminal."""

from __future__ import annotations

from collections.abc import Mapping

from rich.table import Table

from proteus_music.cli.common import console
from proteus_music.library.service import LibraryStats
from proteus_music.models.organization import OrganizationResult
from proteus_music.models.track import Track


def library(stats: LibraryStats) -> None:
    console.print(
        f"[bold]{stats.tracks}[/] canonical track(s) from {stats.source_records} record(s)"
    )
    for source, count in stats.records_by_source.items():
        console.print(f"  {source}: {count} record(s)")
    console.print(f"  with local audio file: {stats.tracks_with_local_file}")
    if stats.tracks_without_analysis_data:
        console.print(
            f"  [yellow]identification-only (excluded from analysis): "
            f"{stats.tracks_without_analysis_data}[/]"
        )


def result(result: OrganizationResult, tracks: Mapping[str, Track], show_tracks: bool) -> None:
    config = result.config
    sizes = f", sizes {config.min_size}..{config.max_size or '∞'}"
    console.print(
        f"Result [bold]{result.id}[/] · {len(result.snapshot_track_ids)} tracks · "
        f"{config.algorithm} (seed {config.seed}{sizes}) · features {result.feature_version}"
    )
    flagged = {a.track_id: a for a in result.low_confidence}
    table = Table(show_edge=False)
    for column in ("#", "Playlist", "Tracks", "Cohesion", "Flagged"):
        table.add_column(column, justify="left" if column == "Playlist" else "right")
    for p in result.playlists:
        low = sum(1 for t in p.track_ids if t in flagged)
        table.add_row(
            str(p.index + 1), p.name, str(len(p.track_ids)), f"{p.cohesion:.2f}", str(low)
        )
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/] {warning}")
    if not show_tracks:
        return
    for p in result.playlists:
        console.print(f"\n[bold]{p.index + 1}. {p.name}[/]\n[dim]{p.description}[/]")
        for track_id in p.track_ids:
            track = tracks[track_id]
            flags = (
                f" [yellow]({', '.join(flagged[track_id].flags)})[/]" if track_id in flagged else ""
            )
            console.print(
                f"  {', '.join(track.artists) or '?'} — {track.title}{flags}", highlight=False
            )
