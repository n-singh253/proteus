"""`proteus enrich ...`: optional, opt-in metadata enrichment from independent sources."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer

from proteus_music.cli.common import EXIT_ERROR, EXIT_PARTIAL, console, database, handled_errors
from proteus_music.integrations.musicbrainz.client import (
    MIN_INTERVAL_S,
    MusicBrainzClient,
    UrllibTransport,
)
from proteus_music.library.enrichment import enrich, plan_enrichment

app = typer.Typer(help="Enrich tracks from independent metadata sources.", no_args_is_help=True)


@app.command("musicbrainz")
def musicbrainz(
    ctx: typer.Context,
    allow_network: Annotated[
        bool, typer.Option("--allow-network", help="Actually contact musicbrainz.org.")
    ] = False,
    contact: Annotated[
        str | None,
        typer.Option(
            envvar="PROTEUS_MUSICBRAINZ_CONTACT",
            help="Your contact URL or email, sent in the User-Agent as MusicBrainz requires.",
        ),
    ] = None,
    max_requests: Annotated[int, typer.Option(min=1, help="Request budget for this run.")] = 300,
    refresh: Annotated[
        bool, typer.Option(help="Ignore the refresh policy; look up again.")
    ] = False,
) -> None:
    """Match tracks to MusicBrainz recordings by ISRC or MusicBrainz ID (CC0 core data only)."""
    with handled_errors():
        conn = database(ctx)
        now = datetime.now(UTC)
        plan = plan_enrichment(conn, now, refresh)
        console.print(
            f"{len(plan.due)} track(s) due for lookup, {plan.fresh} still fresh, "
            f"{plan.without_identifier} without an ISRC or MusicBrainz ID (not looked up)."
        )
        if not allow_network:
            console.print(
                "Dry run: nothing was sent. With --allow-network, only ISRCs or MusicBrainz IDs "
                f"(no titles, artists, files or library data) go to musicbrainz.org, at most one "
                f"request every {MIN_INTERVAL_S:.0f} s and {max_requests} request(s) per run."
            )
            return
        if not contact:
            console.print("[red]MusicBrainz requires contact details: pass --contact.[/]")
            raise typer.Exit(EXIT_ERROR)
        client = MusicBrainzClient(UrllibTransport(), contact, max_requests=max_requests)
        summary = enrich(conn, client, plan, now)
        outcomes = ", ".join(f"{s}: {n}" for s, n in sorted(summary.outcomes.items())) or "none"
        console.print(f"{summary.requests} request(s). Outcomes: {outcomes}.")
        if summary.stopped:
            console.print(
                f"[yellow]Stopped early ({summary.stopped}); {summary.remaining} lookup(s) "
                "remain and resume on the next run.[/]"
            )
            raise typer.Exit(EXIT_PARTIAL)
