"""Optional MusicBrainz enrichment: identifier lookups attached as separate source records.

Each canonical track with an ISRC or MusicBrainz recording ID is looked up at most once per
refresh period. A confident match is stored as its own ``musicbrainz`` source record (CC0
core data, analysis permitted) so existing records and their restrictions never change.
Every outcome is persisted per track, one transaction each, so interrupted runs resume.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from proteus_music.integrations.musicbrainz import matching
from proteus_music.integrations.musicbrainz.client import (
    BudgetExhaustedError,
    MusicBrainzClient,
    MusicBrainzError,
    RateLimitedError,
)
from proteus_music.integrations.musicbrainz.matching import MatchOutcome, MatchStatus
from proteus_music.models.track import Track
from proteus_music.storage import enrichment as lookup_store
from proteus_music.storage import tracks as track_store

# Confirmed matches change rarely; negative results are retried sooner; errors next run.
REFRESH_AFTER = {
    MatchStatus.MATCHED: timedelta(days=180),
    MatchStatus.AMBIGUOUS: timedelta(days=30),
    MatchStatus.NO_MATCH: timedelta(days=30),
    MatchStatus.NOT_FOUND: timedelta(days=30),
    MatchStatus.CONFLICT: timedelta(days=30),
    MatchStatus.ERROR: timedelta(0),
}


@dataclass(frozen=True)
class EnrichmentPlan:
    due: tuple[tuple[Track, str], ...]  # (track, "isrc:..." or "mbid:...")
    fresh: int
    without_identifier: int


@dataclass(frozen=True)
class EnrichmentSummary:
    outcomes: Counter[str]
    requests: int
    remaining: int
    stopped: str | None


def query_for(track: Track) -> str | None:
    mbid = track.musicbrainz_recording_id or next(
        (r.musicbrainz_recording_id for r in track.records if r.musicbrainz_recording_id), None
    )
    if mbid:
        return f"mbid:{mbid}"
    isrc = track.isrc or next((r.isrc for r in track.records if r.isrc), None)
    return f"isrc:{isrc}" if isrc else None


def plan_enrichment(conn: sqlite3.Connection, now: datetime, refresh: bool) -> EnrichmentPlan:
    refresh_at = {} if refresh else lookup_store.refresh_times(conn, matching.PROVIDER)
    due, fresh, without = [], 0, 0
    for track in track_store.load_tracks(conn):
        query = query_for(track)
        if query is None:
            without += 1
        elif track.id in refresh_at and refresh_at[track.id] > now:
            fresh += 1
        else:
            due.append((track, query))
    return EnrichmentPlan(tuple(due), fresh, without)


def _look_up(client: MusicBrainzClient, track: Track, query: str, cache: dict) -> MatchOutcome:
    kind, value = query.split(":", 1)
    if query not in cache:
        if kind == "mbid":
            found = client.recording(value)
            cache[query] = None if found is None else [found]
        else:
            cache[query] = client.recordings_by_isrc(value)
    raw = cache[query]
    recordings = None if raw is None else matching.parse_recordings(raw)
    return matching.match(track, recordings, by_mbid=kind == "mbid")


def _store(
    conn: sqlite3.Connection, track: Track, query: str, outcome: MatchOutcome, now: datetime
):
    with conn:
        recording = outcome.recording
        if recording is not None:
            linked = track_store.find_record(conn, matching.PROVIDER, recording.id)
            if linked is not None and linked[0] != track.id:
                outcome = MatchOutcome(
                    MatchStatus.CONFLICT, None, None, outcome.candidates,
                    f"recording {recording.id} is already linked to track {linked[0]}",
                )  # fmt: skip
            else:
                record = matching.to_record(recording)
                track_store.save_record(conn, track.id, record)
                track_store.fill_missing_fields(conn, track.id, record)
        lookup_store.save_lookup(
            conn,
            lookup_store.LookupRow(
                provider=matching.PROVIDER,
                track_id=track.id,
                query=query,
                status=outcome.status,
                recording_id=recording.id if outcome.status is MatchStatus.MATCHED else None,
                confidence=outcome.confidence,
                candidates=outcome.candidates,
                detail=outcome.detail,
                license=matching.LICENSE,
                fetched_at=now,
                refresh_after=now + REFRESH_AFTER[outcome.status],
            ),
        )
    return outcome.status


def enrich(
    conn: sqlite3.Connection, client: MusicBrainzClient, plan: EnrichmentPlan, now: datetime
) -> EnrichmentSummary:
    outcomes: Counter[str] = Counter()
    cache: dict = {}
    for done, (track, query) in enumerate(plan.due):
        try:
            outcome = _look_up(client, track, query, cache)
        except (RateLimitedError, BudgetExhaustedError) as stop:
            return EnrichmentSummary(outcomes, client.requests, len(plan.due) - done, str(stop))
        except MusicBrainzError as error:
            outcome = MatchOutcome(MatchStatus.ERROR, None, None, 0, str(error))
        outcomes[_store(conn, track, query, outcome, now)] += 1
    return EnrichmentSummary(outcomes, client.requests, 0, None)
