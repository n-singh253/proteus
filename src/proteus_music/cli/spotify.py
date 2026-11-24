"""`proteus spotify ...`: connect your own Spotify app, import, preview and publish."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

import typer

from proteus_music.cli.common import (
    EXIT_ERROR,
    EXIT_PARTIAL,
    MAX_LISTED_ISSUES,
    console,
    database,
    errors,
    handled_errors,
)
from proteus_music.cli.imports import store_batch
from proteus_music.integrations.contract import ExportPreview, MatchStatus
from proteus_music.integrations.http import UrllibTransport
from proteus_music.integrations.spotify import auth
from proteus_music.integrations.spotify.client import SpotifyClient, SpotifyError
from proteus_music.integrations.spotify.export import preview, publish
from proteus_music.integrations.spotify.importer import SOURCE, import_library
from proteus_music.playlists.service import get_result
from proteus_music.storage.tracks import load_tracks, remove_source

app = typer.Typer(
    help="Use your own Spotify development-mode app (requires Spotify Premium).",
    no_args_is_help=True,
)
ATTRIBUTION = "Track data supplied by Spotify (https://open.spotify.com)."
ResultOption = Annotated[str | None, typer.Option("--result", help="Result ID (default: latest).")]
SearchOption = Annotated[
    bool, typer.Option("--search", help="Look up tracks missing from your library by ISRC.")
]


@contextmanager
def _spotify_errors() -> Iterator[None]:
    with handled_errors():
        try:
            yield
        except (SpotifyError, auth.AuthorizationError) as exc:
            errors.print(f"[bold red]Spotify:[/] {exc}")
            raise typer.Exit(EXIT_ERROR) from None


def _client() -> SpotifyClient:
    return SpotifyClient(UrllibTransport(), auth.KeyringStore())


@app.command()
def login(
    client_id: Annotated[
        str,
        typer.Option(envvar="PROTEUS_SPOTIFY_CLIENT_ID", help="Your app's client ID."),
    ],
    redirect_uri: Annotated[
        str, typer.Option(help="Redirect URI registered for your app, exactly.")
    ] = auth.DEFAULT_REDIRECT,
    write: Annotated[
        bool, typer.Option("--write", help="Also allow creating private playlists.")
    ] = False,
    browser: Annotated[bool, typer.Option("--browser/--no-browser")] = True,
) -> None:
    """Authorize Proteus for your Spotify account (PKCE; no client secret is used)."""
    with _spotify_errors():
        auth.validate_redirect(redirect_uri)
        scopes = auth.READ_SCOPES + (auth.WRITE_SCOPES if write else ())
        request = auth.authorization_request(client_id.strip(), redirect_uri, scopes)
        console.print(f"Open this URL to authorize ({', '.join(scopes)}):\n{request.url}")
        if browser:
            auth.open_in_background(request.url)
        code = auth.wait_for_code(redirect_uri, request.state)
        transport = UrllibTransport()
        credentials = auth.exchange_code(transport, client_id.strip(), redirect_uri, request, code)
        auth.KeyringStore().save(credentials)
        console.print(f"Connected; credentials stored in the OS keyring ({len(scopes)} scopes).")


@app.command("import")
def import_(
    ctx: typer.Context,
    saved: Annotated[bool, typer.Option("--saved/--no-saved")] = True,
    playlists: Annotated[bool, typer.Option("--playlists/--no-playlists")] = True,
) -> None:
    """Import saved tracks and your playlists as identification-only records (read-only)."""
    with _spotify_errors():
        batch = import_library(_client(), saved=saved, playlists=playlists)
        store_batch(ctx, batch)
        console.print("Spotify data is used for identity and export only, never for analysis.")
        if any(f.location == SOURCE for f in batch.failures):
            raise typer.Exit(EXIT_PARTIAL)


def _show(plan: ExportPreview, titles: dict[str, str], show_tracks: bool) -> None:
    for playlist in plan.playlists:
        counts = {s: sum(m.status is s for m in playlist.matches) for s in MatchStatus}
        summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
        console.print(f"[bold]{playlist.name}[/] ({summary})", highlight=False)
        problems = [
            m for m in playlist.matches if show_tracks or m.status is not MatchStatus.MATCHED
        ]
        for match in problems[:MAX_LISTED_ISSUES]:
            console.print(
                f"  {match.status:11} {titles[match.track_id]}: {match.reason}", highlight=False
            )
        if len(problems) > MAX_LISTED_ISSUES:
            console.print(f"  ... and {len(problems) - MAX_LISTED_ISSUES} more")
    totals = ", ".join(f"{plan.count(s)} {s}" for s in MatchStatus)
    console.print(f"Total: {totals}. {ATTRIBUTION}")


def _plan(ctx: typer.Context, result_id: str | None, search: bool, show_tracks: bool):
    conn = database(ctx)
    result = get_result(conn, result_id)
    tracks = {t.id: t for t in load_tracks(conn)}
    plan = preview(result, tracks, _client() if search else None)
    _show(
        plan,
        {t: f"{tracks[t].title} — {', '.join(tracks[t].artists)}" for t in tracks},
        show_tracks,
    )
    return plan


@app.command("preview")
def preview_(
    ctx: typer.Context, result_id: ResultOption = None, search: SearchOption = False,
    tracks: Annotated[bool, typer.Option("--tracks", help="List matched tracks too.")] = False,
) -> None:  # fmt: skip
    """Show which tracks of a saved result would be exported. Nothing is written."""
    with _spotify_errors():
        _plan(ctx, result_id, search, tracks)
        console.print("Preview only: nothing was written to Spotify.")


@app.command("publish")
def publish_(
    ctx: typer.Context, result_id: ResultOption = None, search: SearchOption = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm without prompting.")] = False,
) -> None:  # fmt: skip
    """Create private Spotify playlists for a saved result (safe to re-run to resume)."""
    with _spotify_errors():
        plan = _plan(ctx, result_id, search, False)
        n = sum(1 for p in plan.playlists if p.references)
        confirmed = yes or typer.confirm(
            f"Create or resume {n} private playlist(s) with {plan.count(MatchStatus.MATCHED)} "
            "matched track(s) in your Spotify account?"
        )
        if not confirmed:
            console.print("Nothing was written.")
            return
        report = publish(_client(), plan, confirmed=True)
        for item in report.playlists:
            console.print(
                f"{item.name}: {item.added}/{item.planned} — {item.detail}", highlight=False
            )
        if not report.complete:
            console.print("[yellow]Incomplete; re-run the same command to resume safely.[/]")
            raise typer.Exit(EXIT_PARTIAL)


@app.command()
def disconnect(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm without prompting.")] = False,
) -> None:
    """Delete stored Spotify credentials and all Spotify data from the local library."""
    with _spotify_errors():
        conn = database(ctx)
        plan = remove_source(conn, SOURCE, apply=False)
        console.print(
            f"This deletes {plan.records} Spotify record(s): {plan.tracks_removed} track(s) known "
            f"only from Spotify, {plan.results_removed} saved result(s) that include them, and "
            f"Spotify-derived fields of {plan.tracks_rebuilt} other track(s)."
        )
        if not (yes or typer.confirm("Disconnect Spotify and delete this data?")):
            console.print("Nothing was deleted.")
            return
        remove_source(conn, SOURCE, apply=True)
        removed = auth.KeyringStore().delete()
        console.print(f"Disconnected{'' if removed else ' (no stored credentials)'}; data deleted.")
