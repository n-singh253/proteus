# Proteus

Discover the structure of your music.

Proteus is an open-source, local-first music organization engine designed to transform large, unstructured music libraries into cohesive, non-overlapping playlists, without being tied to a single music platform.

## Status

Early development, usable from the command line and installable from source only. Proteus imports a collection, resolves track identity, and represents each track from its metadata. It then organizes every track into exactly one playlist, in user-guided or autonomous mode, and exports the result.

### Implemented

- **Imports:** local audio directories (MP3, FLAC, WAV, AAC, M4A, Ogg Vorbis, Opus, detected by file content rather than extension), CSV with configurable column mapping, JSON, and M3U/M3U8. Imports never modify source files.
- **Identity:** records merge only on a shared ISRC, a shared MusicBrainz recording ID, or an exact normalized title/artist/version match with compatible durations. Live, remix, remaster and other versions stay separate, even when an identifier claims otherwise. Such conflicts are reported at import.
- **Provenance:** every source record keeps its own metadata. Sources imported with `--identification-only` are used for identity and export but never for analysis, and re-importing the same source later cannot lift that restriction. CSV and JSON exports carry an `analysis_permitted` flag that re-import honors. A track is exported as analyzable only if every one of its sources permits analysis. M3U exports omit `#EXTINF` metadata for restricted tracks.
- **Organization:**
  - Modes: an exact playlist count, a count range, required categories, preferred categories, any combination of these, and fully autonomous mode.
  - Optional minimum and maximum playlist sizes.
  - Infeasible requests are rejected with reasons before any heavy computation.
  - Every result is validated independently before it is saved, shown or exported, and saved with its configuration, seed, feature version, quality metrics, count-selection trace and category evidence.
- **Local web interface:** `proteus serve` covers import, organization, playlist and flagged-track inspection, and export, using the same services as the CLI. See [Web interface](#web-interface).
- **Acoustic analysis:** CPU-only loudness, spectral, timbre (MFCC) and tempo features from permitted local audio, cached by file identity, fused with metadata. See [Acoustic analysis](#acoustic-analysis).
- **Duplicate candidates:** Chromaprint-style fingerprints list possible duplicate recordings for review; nothing is merged automatically.
- **Enrichment (optional, opt-in):** MusicBrainz lookups by ISRC or MusicBrainz recording ID, using CC0 core data only. See [Metadata enrichment](#metadata-enrichment).
- **Exports:** M3U8 (one file per playlist, absolute or relative paths), CSV, and JSON (which can be re-imported). Tracks that cannot be exported are reported, and existing files are never overwritten unless requested.

### Planned

Audio embeddings, reviewed duplicate merging, title/artist search for enrichment, additional clustering algorithms (density- and graph-based), visual exploration and manual playlist editing in the web interface, authorized streaming-service adapters (Spotify, Apple Music, YouTube), and packaged releases. Platform integrations are subject to API availability, authorization, and applicable data-use policies.

## Install from source

Requires Python 3.11+, a Rust toolchain, and [uv](https://docs.astral.sh/uv/).

```sh
uv sync          # builds the Rust extension and installs dependencies into .venv
uv run proteus --help
```

## Usage

```sh
proteus import local ~/Music                     # scan recursively (--no-recursive to disable)
proteus import csv tracks.csv --map title="Song Name" --separator ";"
proteus import json library.json
proteus import m3u playlist.m3u8
proteus library                                  # track and source counts

proteus organize --playlists 10                                    # exact count
proteus organize --min-playlists 5 --max-playlists 15              # best count in a range
proteus organize --categories "Hip-Hop,R&B,House,Rock"             # one playlist per category
proteus organize --playlists 10 --categories "Hip-Hop,Rock,House"  # 3 categories + 7 discovered
proteus organize --mode auto                                       # choose count and groups
proteus organize --min-playlists 5 --max-playlists 10 --min-size 20 --max-size 500 \
    --categories "Hip-Hop,House" --prefer-categories "Chill,Workout"
proteus report --tracks                          # latest result; pass an ID for an older one
proteus results                                  # list saved results

proteus export --target m3u8 --out ./playlists --relative
proteus export --target csv --out ./playlists
```

`--categories` are hard requirements: each one becomes exactly one playlist with its name preserved. `--prefer-categories` become playlists only when the library strongly supports them and a slot is free. Every organize run reports the chosen count and why, category evidence, quality metrics, constraint satisfaction, and warnings. Add `--json` for machine-readable output, `--seed` for a different reproducible run, and `--algorithm spherical-kmeans` to use the plain baseline.

The library database defaults to the per-user data directory. Override it with `--db PATH` or `PROTEUS_DB`. Exit codes: `1` error or invalid configuration, `2` infeasible constraints, `3` partial export.

If a CSV or JSON file comes from a provider whose terms restrict analysis or machine-learning use of its data, import it with `--identification-only`.

## Web interface

```sh
npm --prefix frontend ci && npm --prefix frontend run build   # once, from a source checkout
proteus serve                                                # opens http://localhost:8765/
```

The page covers the core workflow:
1. Import a CSV, JSON or M3U file, or a local audio folder.
2. Organize in autonomous mode or with an exact count, a range or categories, plus optional size limits and a feature set.
3. Inspect playlists, their descriptions and flagged tracks.
4. Export the selected saved result as M3U8, CSV or JSON.

Imports, organization and exports run as background jobs with stage progress, cancellation and actionable error messages. Infeasible requests list their reasons. Results are written in one transaction at the end, so a failed or cancelled job saves nothing. The CLI and the web interface read the same database and the same saved results.

**Protections:**
- The server binds to `127.0.0.1` and refuses other interfaces unless you pass `--allow-remote`.
- Every API request must carry a per-session token that is embedded only in the served page.
- The `Host` header must name the server, which defeats DNS rebinding, and cross-origin requests are rejected.
- Paths typed into the UI must be absolute and inside the allowed folders (your home directory by default; change with `--allow-root`). Import paths must be existing files of the right type or existing folders. Exports never overwrite unless you tick the box.

## Acoustic analysis

```sh
proteus import local ~/Music                     # or CSV/JSON/M3U entries with file paths
proteus analyze audio                            # decode and describe permitted files (cached)
proteus organize --mode auto                     # fused metadata + audio (the default)
proteus organize --mode auto --features metadata # or: audio
proteus analyze duplicates                       # fingerprint duplicate candidates, review only
```

Only files referenced by analysis-permitted source records are decoded, and they are never modified. Decoding runs natively, in parallel, in batches of 16 files (`--batch-size`, `--threads`). Each file contributes a mono excerpt of at most 30 s (`--excerpt-seconds`), taken from the middle of longer tracks.

Every outcome is cached in SQLite, keyed by path, size, modification time and extractor version, so unchanged files are never decoded twice. Decode failures are cached too, so they aren't retried until the file changes.

**Codecs, verified with real encodes:**
- Decoded: MP3, FLAC, WAV/PCM, AAC (M4A and ADTS), ALAC, Ogg Vorbis.
- Not decoded: Opus (Symphonia has no Opus decoder) and CAF containers. These are reported as unanalyzable.

**Features (`acoustic-v1`):**
- RMS loudness (dBFS, not LUFS) and loudness range.
- Spectral centroid mean and spread, bandwidth, 85% rolloff and flatness, all computed over 0–11.025 kHz so files at different sample rates compare.
- 13 mean MFCCs over 40 mel bands (timbre).
- Tempo from a spectral-flux onset autocorrelation. It is reported only when onsets are pronounced and the periodicity is strong, so noise, pads and steady tones have no tempo rather than a guessed one. On synthetic click tracks it was within 1.5% from 80 to 174 BPM.

**Fusion:**
- Acoustic features are z-scored across the library's analyzed tracks and form a block weighted 0.30 next to the metadata blocks. Tempo is a separate 0.05 block, a soft log-tempo histogram.
- Availability masks mean a track without analyzed audio contributes no acoustic evidence, and its other blocks are renormalized.
- `--features metadata` reproduces the metadata-only results exactly, and so does `fused` when no audio has been analyzed. Category names are matched against metadata, so categories need `fused` or `metadata`.
- Playlist descriptions add a separate "Measured audio" phrase (median tempo, loudness and spectral centroid), distinct from the metadata-derived genres and tags.

`benchmarks/acoustic.py` compares feature sets on a synthetic collection: four synthesized styles, AAC/FLAC/WAV, 25% wrong genres, 30% missing genres and tags, and 15% of tracks without audio. On 400 tracks:

| Feature set | ARI, exact 4 | ARI, auto | Auto playlists |
| --- | --- | --- | --- |
| metadata | 0.51 | 0.51 | 4 |
| audio | 0.69 | 0.50 | 9 (over-split) |
| fused | 0.82–0.84 | 0.82–0.84 | 4 |

Cold analysis of 344 files (30 s each) took 3.5 s on an Apple M5 Pro; the warm run took 0.01 s. These are synthetic sounds: they test mechanics, not listening quality. Audio alone over-splits in auto mode, so fused is the default.

**Duplicate candidates:** `benchmarks/fingerprints.py` compares a synthetic song against variants of it.
- Same recording: re-encodes and level or rate changes score 0.96–1.00.
- **A remaster scores 0.95**, so it is indistinguishable from a lossy re-encode.
- A radio edit scores 0.77, an extended mix 0.56, a live performance 0.30 and an unrelated song 0.18.

Because no threshold separates a remaster from a copy, fingerprints never merge tracks. `proteus analyze duplicates` lists pairs above 0.6 with a note when their version descriptors differ, for you to review. The identity rules stay as they were.

## Metadata enrichment

Proteus can add independently licensed metadata from [MusicBrainz](https://musicbrainz.org). This is the only workflow that uses the network, and it is off unless you ask for it:

```sh
proteus enrich musicbrainz                       # dry run: shows what would be looked up, sends nothing
proteus enrich musicbrainz --allow-network --contact "https://example.org/you-or-your-project"
proteus organize --mode auto                     # re-organize with the enriched metadata
```

**What is sent:** only each track's ISRC or MusicBrainz recording ID, plus a User-Agent with the contact you give (`--contact` or `PROTEUS_MUSICBRAINZ_CONTACT`), as MusicBrainz requires. No titles, artists, files or library contents are sent. Tracks without either identifier are not looked up.

**What is used:** only MusicBrainz core data, which is CC0: recording ID, title, artist credit, length, ISRCs and release dates. These provide artists and the earliest release year for analysis, plus identifiers. Tags and genre associations are supplementary data under CC BY-NC-SA, so they are never used. This means enrichment helps sparse tracks join playlists by artist and era, but adds no genres.

**Matching:** a recording is accepted only when exactly one candidate agrees with the track on title, version (live, remix, remaster and so on, including MusicBrainz disambiguation), artist when known, and duration within 3 s when known.
- Several consistent candidates count as *ambiguous*; inconsistent ones as *no-match*; an unknown identifier as *not-found*.
- A recording already linked to another track is a *conflict*. All of these leave the track unchanged.
- Each match is stored as a separate `musicbrainz` source record, so existing records and their identification-only restrictions are never modified.

**Requests and resuming:**
- Requests are spaced at least 1 s apart, and each run has a request budget (`--max-requests`, default 300).
- HTTP 503 (throttling) and transient failures are retried up to 3 times with backoff; persistent throttling stops the run.
- Every outcome is stored in SQLite with its query, confidence, license and refresh time, so an interrupted run resumes where it left off.
- Matches are refreshed after 180 days, unmatched results after 30, and errors on the next run. `--refresh` forces new lookups.

## How organization works

All modes share one pipeline.

1. **Representation:** each track becomes a weighted TF-IDF vector over genres, tags, artists and decade. The weights are experimental defaults. Blocks a track lacks are renormalized away instead of being treated as zeros. Genres and tags are the core modalities. Tracks with only an artist or year are matched to playlists but don't shape them (flag `partial-features`).
2. **Categories:** a category name is matched against the library's own permitted genre, tag and decade vocabulary, first as a whole phrase, then word by word. This yields a prototype in the same vector space as tracks.
   - Evidence is the number of tracks with compatibility ≥ 0.15.
   - A required category with no evidence is rejected rather than filled with unrelated songs.
   - Few supporting tracks, or a partial word match (such as "Late Night Driving" matching only "night"), is reported as weak.
3. **Prototypes:** `kmeans-merge` (the default) fits k-means on unit vectors into about 2k + 8 micro-clusters and merges them by average linkage. Plain k-means prefers splitting a large group over isolating a small distinct one, and merging fixes that. With categories, discovered groups are seeded from tracks that match no category well. A joint refinement then reassigns every track by its overall best fit, with category prototypes anchored to their meaning.
4. **Count selection:** used for ranges and auto mode.
   - Candidate counts are evaluated on a seeded sample of up to 6,000 tracks, with size constraints scaled to the sample.
   - Small ranges are searched exhaustively. Otherwise the smallest 7 counts are searched densely, then a geometric grid refined around the best, for at most 14 evaluations.
   - In auto mode the range adapts to the library: up to 40 playlists, no more than the library can fill at a practical size, and no more than twice the number of well-supported genre/tag terms.
5. **Assignment:** each track goes to the playlist that maximizes total similarity under the size bounds. This is an exact transportation LP solved with HiGHS, which is integral because its constraint matrix is totally unimodular. Column generation starts from each track's top 4 playlists and adds variables until LP duality proves global optimality. MILP remains as a fallback.
6. **Missing information:** tracks with no permitted metadata go to an explicit `Unclassified` playlist when the count is flexible and size limits allow. Otherwise they get a deterministic fallback and the `missing-features` flag. Low-similarity, ambiguous and constraint-displaced tracks are flagged too.
7. **Naming:** user category names are kept verbatim. Discovered playlists are named after their most distinctive genres or tags, then artists or decades. Near-duplicate names that differ only in punctuation or word order are avoided.

### Quality scoring

Results report clustering quality and practical usefulness separately:

- **Clustering quality:** sampled cosine silhouette, cohesion, centroid separation, and assignment compatibility.
- **Practical usefulness:** playlist sizes, fragmentation, low-confidence rate, semantic coherence (the share of tracks carrying their playlist's most common genre or tag), and coverage.

Count selection maximizes `silhouette − 0.5 × fragmentation − 0.25 × low-confidence rate`. Fragmentation is the share of tracks in playlists smaller than the practical size of `max(15, min(25, n/40))` tracks, about an hour of music. A single playlist scores 0.12, so a split must show at least that much structure. Ties go to fewer playlists. Unbalanced sizes are not penalized: only the share of tracks in undersized playlists is.

These constants were set empirically on the synthetic libraries in `benchmarks/quality.py` and are starting points, not universal truths. Silhouette favors the finer level of nested structures, and when the data is genuinely ambiguous, several counts are defensible.

## Benchmarks

```sh
uv run python benchmarks/quality.py                      # auto mode vs known synthetic groups
uv run python benchmarks/performance.py --sizes 1000 10000 50000 100000
```

Measured on an Apple M5 Pro with Python 3.14, synthetic metadata in 20 imbalanced groups, and the default algorithm. Each case runs in a fresh process. There is no feature cache yet, so all runs are cold.

| Tracks | Fixed 15 | Fixed 15, sizes bound | Range 5–25 | 5 categories | 5 categories + 10 discovered | Auto | Auto, max size n/8 | Peak RAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 0.07 s | 0.12 s | 0.47 s | 0.02 s | 0.06 s | 0.48 s | 0.78 s | ≤ 0.2 GB |
| 10,000 | 0.30 s | 0.75 s | 1.6 s | 0.17 s | 0.25 s | 1.6 s | 4.2 s | ≤ 0.3 GB |
| 50,000 | 1.1 s | 3.1 s | 3.2 s | 0.70 s | 0.92 s | 3.2 s | 6.4 s | ≤ 0.6 GB |
| 100,000 | 2.3 s | 6.1 s | 5.3 s | 1.4 s | 1.9 s | 5.4 s | 11.6 s | ≤ 0.93 GB |

Count selection costs about 3 s at 100k tracks because it runs on a 6,000-track sample, so it grows slowly with library size. Constrained count selection is the slowest stage, because every candidate count solves a constrained assignment on the sample.

In auto mode on the synthetic libraries:
- Clear 4- and 8-group, noisy, small 2-group and 30%-missing libraries recover the generating count in every seed.
- The imbalanced 600/200/80/40/20 library recovers all 5 groups (plain k-means finds only 3).
- The hierarchical 3 × 3 library chooses the 9 subgroups.
- Structureless and tiny (12-track) libraries get a single playlist.
- At 60% metadata noise the generating groups are barely present, and the chosen count (8–9) is not reliable.

## Architecture

| Path | Responsibility |
| --- | --- |
| `crates/proteus-core` | Rust extension `proteus_music._native`: parallel scanning, content-based format detection, tag extraction (PyO3, Lofty, Rayon). |
| `src/proteus_music/models` | Normalized track, organization and quality models. |
| `src/proteus_music/library` | Identity resolution and import service. |
| `src/proteus_music/integrations` | Adapter contract plus local and portable-file adapters. |
| `src/proteus_music/engine` | Features, categories, clustering, count planning, assignment, evaluation and naming. Independent of providers and interfaces. |
| `src/proteus_music/playlists` | Organization and export services shared by all interfaces. |
| `src/proteus_music/storage` | SQLite persistence with versioned migrations. |
| `src/proteus_music/cli` | Thin Typer CLI. |
| `src/proteus_music/api` | FastAPI app for `proteus serve`: local-only protections, jobs, typed views. |
| `frontend/` | React/TypeScript interface, built into `src/proteus_music/api/static/` (not committed). |
| `benchmarks/` | Synthetic-library quality and performance benchmarks (not part of the package). |

## Limitations

- Enrichment looks up identifiers only; tracks with neither an ISRC nor a MusicBrainz ID cannot be enriched yet. Release years come from the at most 25 releases MusicBrainz returns per recording.
- Acoustic features are conventional signal statistics, not learned embeddings, and no listening review has been done yet. Opus files cannot be decoded.
- Organization uses metadata (genres, tags, artists, decade) plus conventional audio statistics when analyzed. Categories work only as well as the library's genre and tag vocabulary; there are no synonyms ("Chill" does not match "calm").
- Count selection is heuristic and was tuned on synthetic data. It has not yet been evaluated on real libraries or with listeners.
- With only required categories and no count, every track must land in one of them. Tracks that fit none are flagged low-similarity instead of being left out.

## Contributing

Run the same checks as CI before submitting changes:

```sh
uv run ruff format --check && uv run ruff check && uv run pytest
cargo fmt --all --check && cargo clippy --all-targets -- -D warnings
npm --prefix frontend ci && npm --prefix frontend run build   # strict type check + bundle
```

After changing Rust code, `uv sync` rebuilds the extension. Tests use small synthetic fixtures only. Never commit music, databases, credentials or model weights.

## License

Apache-2.0. Third-party data, models, dependencies, and provider content retain their respective licenses and restrictions.
