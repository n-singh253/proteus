"""`proteus import ...` commands. Imports never modify the source files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from proteus_music.cli.common import MAX_LISTED_ISSUES, console, database, handled_errors
from proteus_music.integrations.contract import ImportBatch
from proteus_music.integrations.files import csv_format, json_format, m3u
from proteus_music.integrations.local import scanner
from proteus_music.library.service import import_batch

app = typer.Typer(help="Import a music collection into the local library.", no_args_is_help=True)

ExistingFile = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]
IdentificationOnly = Annotated[
    bool,
    typer.Option(
        "--identification-only",
        help="Use this source only for identity and export, never for analysis "
        "(e.g. files exported from a provider whose terms restrict analysis).",
    ),
]


def store_batch(ctx: typer.Context, batch: ImportBatch) -> None:
    summary = import_batch(database(ctx), batch)
    console.print(
        f"[bold]{summary.source}[/]: read {summary.records_read} record(s) — "
        f"{summary.new_tracks} new track(s), {summary.matched_existing} matched existing, "
        f"{summary.refreshed_records} refreshed."
    )
    for label, issues in (("skipped", summary.failures), ("warning", summary.warnings)):
        if issues:
            console.print(f"[yellow]{len(issues)} {label}:[/]")
            for issue in issues[:MAX_LISTED_ISSUES]:
                console.print(f"  {issue.location}: {issue.reason}", highlight=False)
            if len(issues) > MAX_LISTED_ISSUES:
                console.print(f"  ... and {len(issues) - MAX_LISTED_ISSUES} more")


@app.command("local")
def import_local(
    ctx: typer.Context,
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    recursive: Annotated[bool, typer.Option("--recursive/--no-recursive")] = True,
    threads: Annotated[int | None, typer.Option(min=1, help="Maximum scanner threads.")] = None,
) -> None:
    """Scan a directory of audio files and read their embedded metadata."""
    with handled_errors():
        store_batch(
            ctx, scanner.read_directory(directory, recursive=recursive, max_threads=threads)
        )


@app.command("csv")
def import_csv(
    ctx: typer.Context,
    path: ExistingFile,
    mapping: Annotated[
        list[str] | None,
        typer.Option("--map", help="Map a field to a column, e.g. --map title='Song Name'."),
    ] = None,
    separator: Annotated[str, typer.Option(help="Separator inside multi-value cells.")] = ";",
    encoding: Annotated[str, typer.Option(help="Text encoding of the file.")] = "utf-8-sig",
    identification_only: IdentificationOnly = False,
) -> None:
    """Import a CSV track list."""
    with handled_errors():
        overrides = {}
        for item in mapping or []:
            field, sep, column = item.partition("=")
            if not sep:
                raise ValueError(f"--map expects field=Column, got {item!r}")
            overrides[field.strip()] = column
        batch = csv_format.read_csv(
            path,
            column_overrides=overrides,
            separator=separator,
            encoding=encoding,
            analysis_permitted=not identification_only,
        )
        store_batch(ctx, batch)


@app.command("json")
def import_json(
    ctx: typer.Context, path: ExistingFile, identification_only: IdentificationOnly = False
) -> None:
    """Import a JSON track list or a Proteus JSON export."""
    with handled_errors():
        store_batch(ctx, json_format.read_json(path, analysis_permitted=not identification_only))


@app.command("m3u")
def import_m3u(
    ctx: typer.Context, path: ExistingFile, identification_only: IdentificationOnly = False
) -> None:
    """Import an M3U/M3U8 playlist, reading tags from referenced local files."""
    with handled_errors():
        store_batch(ctx, m3u.read_m3u(path, analysis_permitted=not identification_only))
