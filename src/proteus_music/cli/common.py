"""Shared CLI plumbing: database access and error-to-exit-code translation."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console

from proteus_music.config import default_database_path
from proteus_music.engine.optimization.assignment import InfeasibleConstraintsError
from proteus_music.storage.db import connect

console = Console(soft_wrap=True)
errors = Console(stderr=True, soft_wrap=True)

EXIT_ERROR = 1
EXIT_INFEASIBLE = 2
EXIT_PARTIAL = 3
MAX_LISTED_ISSUES = 20


def database(ctx: typer.Context) -> sqlite3.Connection:
    path: Path = ctx.obj or default_database_path()
    return connect(path)


@contextmanager
def handled_errors() -> Iterator[None]:
    """Print expected failures as actionable messages instead of tracebacks."""
    try:
        yield
    except InfeasibleConstraintsError as exc:
        errors.print("[bold red]Infeasible configuration:[/]")
        for reason in exc.reasons:
            errors.print(f"  - {reason}")
        raise typer.Exit(EXIT_INFEASIBLE) from None
    except (LookupError, ValueError, OSError) as exc:
        errors.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(EXIT_ERROR) from None
