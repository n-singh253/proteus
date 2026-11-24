"""Human-readable rendering of results for the terminal."""

from __future__ import annotations

from collections.abc import Mapping

from rich.table import Table

from proteus_music.cli.common import console
from proteus_music.library.service import LibraryStats
from proteus_music.models.organization import OrganizationConfig, OrganizationResult
from proteus_music.models.quality import CountSelection, QualityReport
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


def _mode(config: OrganizationConfig) -> str:
    if config.mode == "auto":
        return "auto"
    if config.playlists is not None:
        kind = f"exact {config.playlists}"
    elif config.has_range:
        kind = f"range {config.min_playlists or 1}..{config.max_playlists or 'auto'}"
    else:
        kind = "one per category"
    return f"guided, {kind}"


def _selection(selection: CountSelection) -> None:
    console.print(f"Playlist count {selection.chosen}: {selection.reason}.", highlight=False)
    tried = "  ".join(f"{c.playlists}→{c.score:.3f}" for c in selection.evaluated)
    console.print(f"  [dim]evaluated on {selection.sample_size} tracks: {tried}[/]")


def _quality(q: QualityReport) -> None:
    silhouette = "n/a" if q.silhouette is None else f"{q.silhouette:.3f}"
    separation = "n/a" if q.separation is None else f"{q.separation:.2f}"
    console.print(
        f"Clustering: silhouette {silhouette} · cohesion {q.cohesion:.2f} · separation "
        f"{separation} · compatibility {q.compatibility:.2f}",
        highlight=False,
    )
    console.print(
        f"Usefulness: sizes {q.smallest_playlist}..{q.largest_playlist} · "
        f"{q.fragmentation:.0%} in undersized playlists · {q.low_confidence_rate:.0%} "
        f"low-confidence · semantic coherence {q.semantic_coherence:.0%} · "
        f"{q.analyzable_share:.0%} with genre/tag metadata · score {q.score:.3f}",
        highlight=False,
    )


def _constraints(result: OrganizationResult) -> None:
    config = result.config
    low, high = config.count_bounds()
    required = len(config.required_categories)
    console.print(
        f"[green]Constraints satisfied (validated):[/] {len(result.playlists)} playlists within "
        f"{low}..{high or 'any'} · sizes {config.min_size}..{config.max_size or 'any'} · "
        f"{required}/{required} required categories · every track assigned exactly once",
        highlight=False,
    )


def _categories(result: OrganizationResult) -> None:
    table = Table(title="Categories", show_edge=False, title_justify="left")
    for column in ("Category", "Kind", "Evidence", "Support", "Matched terms", "Note"):
        table.add_column(column)
    for c in result.categories:
        kind = "required" if c.required else ("preferred" if c.kept else "preferred (dropped)")
        table.add_row(c.name, kind, c.evidence, str(c.supporting_tracks),
                      ", ".join(c.matched_terms) or "-", c.note)  # fmt: skip
    console.print(table)


def result(result: OrganizationResult, tracks: Mapping[str, Track], show_tracks: bool) -> None:
    config = result.config
    console.print(
        f"Result [bold]{result.id}[/] · {_mode(config)} · {len(result.snapshot_track_ids)} tracks "
        f"· {config.algorithm} (seed {config.seed}) · features {result.feature_version}"
    )
    if result.selection:
        _selection(result.selection)
    if result.categories:
        _categories(result)
    flagged = {a.track_id: a for a in result.low_confidence}
    table = Table(show_edge=False)
    for column in ("#", "Playlist", "Origin", "Tracks", "Cohesion", "Flagged"):
        table.add_column(column, justify="left" if column in ("Playlist", "Origin") else "right")
    for p in result.playlists:
        low = sum(1 for t in p.track_ids if t in flagged)
        table.add_row(str(p.index + 1), p.name, p.origin, str(len(p.track_ids)),
                      f"{p.cohesion:.2f}", str(low))  # fmt: skip
    console.print(table)
    if result.quality:
        _quality(result.quality)
    _constraints(result)
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/] {warning}", highlight=False)
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
