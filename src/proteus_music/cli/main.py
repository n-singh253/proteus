"""`proteus` command-line entry point."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from proteus_music.cli import imports, render
from proteus_music.cli.common import (
    EXIT_PARTIAL,
    MAX_LISTED_ISSUES,
    console,
    database,
    handled_errors,
)
from proteus_music.integrations.contract import ExportOptions
from proteus_music.library.service import library_stats
from proteus_music.models.organization import Category, OrganizationConfig, OrganizationMode
from proteus_music.playlists import service
from proteus_music.storage import results as result_store
from proteus_music.storage.tracks import load_tracks

app = typer.Typer(
    help="Proteus: organize a music library into cohesive, non-overlapping playlists.",
    no_args_is_help=True,
)
app.add_typer(imports.app, name="import")

JsonFlag = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]


@app.callback()
def main(
    ctx: typer.Context,
    db: Annotated[
        Path | None,
        typer.Option(help="Library database path (default: $PROTEUS_DB or user data dir)."),
    ] = None,
) -> None:
    ctx.obj = db


@app.command()
def library(ctx: typer.Context, as_json: JsonFlag = False) -> None:
    """Show library statistics."""
    with handled_errors():
        stats = library_stats(database(ctx))
        if as_json:
            print(json.dumps(asdict(stats), indent=2))
        else:
            render.library(stats)


@app.command()
def sources() -> None:
    """List available source and destination adapters and their capabilities."""
    for adapter in service.ADAPTERS:
        caps = ", ".join(sorted(adapter.capabilities))
        console.print(f"[bold]{adapter.name}[/] ({caps}): {adapter.description}", highlight=False)


def _categories(names: str | None, required: bool) -> list[Category]:
    return [Category(name=n, required=required) for n in (names or "").split(",") if n.strip()]


@app.command()
def organize(
    ctx: typer.Context,
    playlists: Annotated[
        int | None, typer.Option("--playlists", "-n", min=1, help="Exact playlist count.")
    ] = None,
    min_playlists: Annotated[
        int | None, typer.Option(min=1, help="Range: fewest playlists.")
    ] = None,
    max_playlists: Annotated[int | None, typer.Option(min=1, help="Range: most playlists.")] = None,
    categories: Annotated[
        str | None, typer.Option(help='Required categories, comma-separated: "Hip-Hop,R&B".')
    ] = None,
    prefer_categories: Annotated[
        str | None, typer.Option(help="Preferred categories, created only when supported.")
    ] = None,
    mode: Annotated[
        OrganizationMode, typer.Option(help="guided (count/categories given) or auto.")
    ] = OrganizationMode.GUIDED,
    min_size: Annotated[int, typer.Option(min=1, help="Minimum tracks per playlist.")] = 1,
    max_size: Annotated[
        int | None, typer.Option(min=1, help="Maximum tracks per playlist.")
    ] = None,
    algorithm: Annotated[
        str, typer.Option(help="Clustering algorithm: kmeans-merge or spherical-kmeans.")
    ] = "kmeans-merge",
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility.")] = 0,
    as_json: JsonFlag = False,
) -> None:
    """Organize the whole library into exclusive playlists and save the result."""
    with handled_errors():
        config = OrganizationConfig(
            mode=mode,
            playlists=playlists,
            min_playlists=min_playlists,
            max_playlists=max_playlists,
            categories=(*_categories(categories, True), *_categories(prefer_categories, False)),
            min_size=min_size,
            max_size=max_size,
            algorithm=algorithm,
            seed=seed,
        )
        conn = database(ctx)
        result = service.organize_library(conn, config)
        if as_json:
            print(result.model_dump_json(indent=2))
        else:
            render.result(result, {}, show_tracks=False)
            console.print(f"Saved. Inspect with `proteus report {result.id} --tracks`.")


@app.command()
def results(ctx: typer.Context) -> None:
    """List saved organization results, newest first."""
    with handled_errors():
        for result_id, created, n_playlists, n_tracks in result_store.list_results(database(ctx)):
            console.print(
                f"{result_id}  {created:%Y-%m-%d %H:%M}  {n_playlists} playlists, {n_tracks} tracks"
            )


@app.command()
def report(
    ctx: typer.Context,
    result_id: Annotated[str | None, typer.Argument(help="Result ID (default: latest).")] = None,
    tracks: Annotated[bool, typer.Option("--tracks", help="List every track.")] = False,
    as_json: JsonFlag = False,
) -> None:
    """Inspect an organization result."""
    with handled_errors():
        conn = database(ctx)
        result = service.get_result(conn, result_id)
        if as_json:
            print(result.model_dump_json(indent=2))
            return
        render.result(result, {t.id: t for t in load_tracks(conn)}, show_tracks=tracks)


@app.command()
def export(
    ctx: typer.Context,
    target: Annotated[str, typer.Option(help=f"One of: {', '.join(service.EXPORTERS)}.")],
    out: Annotated[Path, typer.Option(help="Output directory.")],
    result_id: Annotated[
        str | None, typer.Option("--result", help="Result ID (default: latest).")
    ] = None,
    relative: Annotated[
        bool, typer.Option(help="Write paths relative to the output directory.")
    ] = False,
    overwrite: Annotated[bool, typer.Option(help="Replace existing export files.")] = False,
) -> None:
    """Export an organization result to portable files."""
    with handled_errors():
        options = ExportOptions(relative_paths=relative, overwrite=overwrite)
        report_ = service.export_result(
            database(ctx), target, out, result_id=result_id, options=options
        )
        for path in report_.files:
            console.print(f"wrote {path}", highlight=False)
        console.print(f"{report_.exported_tracks} track(s) exported.")
        if not report_.complete:
            console.print(f"[yellow]{len(report_.unexportable)} track(s) could not be exported:[/]")
            for issue in report_.unexportable[:MAX_LISTED_ISSUES]:
                console.print(f"  {issue.reason}", highlight=False)
            if len(report_.unexportable) > MAX_LISTED_ISSUES:
                console.print(f"  ... and {len(report_.unexportable) - MAX_LISTED_ISSUES} more")
            console.print(
                "The saved result still contains every track; only this export is partial."
            )
            raise typer.Exit(EXIT_PARTIAL)
