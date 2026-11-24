"""`proteus serve`: the local web interface."""

from __future__ import annotations

import secrets
import threading
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from proteus_music.cli.common import EXIT_ERROR, errors
from proteus_music.config import default_database_path


def serve(
    ctx: typer.Context,
    host: Annotated[
        str, typer.Option(help="Interface to bind (loopback by default).")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
    browser: Annotated[bool, typer.Option("--browser/--no-browser")] = True,
    allow_root: Annotated[
        list[Path] | None,
        typer.Option(help="Folder the UI may read and write (repeatable; default: home)."),
    ] = None,
    allow_remote: Annotated[
        bool, typer.Option(help="Permit a non-loopback host (exposes your library).")
    ] = False,
) -> None:
    """Start the local web interface and open it in your browser."""
    import uvicorn

    from proteus_music.api.app import create_app
    from proteus_music.api.security import LOOPBACK_HOSTS, allowed_hosts

    if host not in LOOPBACK_HOSTS and not allow_remote:
        errors.print(f"[red]Refusing to bind {host}: use a loopback host or --allow-remote.[/]")
        raise typer.Exit(EXIT_ERROR)
    roots = [p.expanduser().resolve() for p in (allow_root or [Path.home()])]
    app = create_app(
        db_path=ctx.obj or default_database_path(),
        token=secrets.token_urlsafe(32),
        hosts=allowed_hosts(host, port),
        roots=roots,
    )
    url = f"http://{'localhost' if host in LOOPBACK_HOSTS else host}:{port}/"
    typer.echo(f"Proteus is running at {url} (Ctrl+C to stop). Allowed folders: "
               + ", ".join(map(str, roots)))  # fmt: skip
    if browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
