# Proteus

Discover the structure of your music.

Proteus is an open-source, local-first music organization engine designed to transform large, unstructured music libraries into cohesive, non-overlapping playlists, without being tied to a single music platform.

## Status

Early development. A command-line vertical slice works end to end: import a collection, resolve track identity, build a metadata representation, organize every track into exactly one playlist under hard constraints, and export the result. It is installable from source only.

### Implemented

- **Imports:** local audio directories (MP3, FLAC, WAV, AAC, M4A, Ogg Vorbis, Opus, detected by file content rather than extension), CSV with configurable column mapping, JSON, and M3U/M3U8. Imports never modify source files.
- **Identity:** records merge only on a shared ISRC, a shared MusicBrainz recording ID, or an exact normalized title/artist/version match with compatible durations. Live, remix, remaster and other versions stay separate.
- **Provenance:** every source record keeps its own metadata. Sources imported with `--identification-only` are used for identity and export but never for analysis.
- **Organization:** an exact playlist count with optional minimum and maximum playlist sizes. Infeasible requests are rejected with reasons. Every result is validated independently before it is saved or exported, and saved with its configuration, seed and feature version.
- **Exports:** M3U8 (one file per playlist, absolute or relative paths), CSV, and JSON (which can be re-imported). Tracks that cannot be exported are reported, and existing files are never overwritten unless requested.

### Planned

Playlist count ranges and autonomous count selection, user-defined categories, acoustic features and audio embeddings, fingerprint-based duplicate detection, metadata enrichment, the local web interface (`proteus serve`), authorized streaming-service adapters (Spotify, Apple Music, YouTube), and packaged releases. Platform integrations are subject to API availability, authorization, and applicable data-use policies.

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

proteus organize --playlists 10                  # exact count
proteus organize -n 10 --min-size 20 --max-size 400 --seed 1
proteus report --tracks                          # latest result; pass an ID for an older one
proteus results                                  # list saved results

proteus export --target m3u8 --out ./playlists --relative
proteus export --target csv --out ./playlists
```

The library database defaults to the per-user data directory. Override it with `--db PATH` or `PROTEUS_DB`. Exit codes: `1` error, `2` infeasible constraints, `3` partial export.

If a CSV or JSON file comes from a provider whose terms restrict analysis or machine-learning use of its data, import it with `--identification-only`.

## How organization works

1. **Representation:** each track becomes a weighted TF-IDF vector over genres, tags, artists and decade. The weights are experimental defaults. Blocks a track lacks are renormalized away instead of being treated as zeros. Tracks with no analyzable metadata are flagged and placed by an explicit fallback.
2. **Clustering:** spherical k-means on unit vectors produces one prototype per playlist. Algorithms are interchangeable behind a small interface.
3. **Assignment:** each track goes to the playlist that maximizes total similarity. When size bounds bind, Proteus solves the transportation problem exactly as a linear program with HiGHS. The problem is totally unimodular, so the solution is integral.
4. **Validation and naming:** results are checked for exclusive, complete assignment and for the requested constraints. Playlists are named from their most distinctive metadata. Low-similarity, ambiguous and constraint-displaced tracks are flagged.

On an Apple M5 Pro with synthetic metadata, organizing 50,000 tracks into 15 playlists took about 1.6 s unconstrained and 6.8 s with binding size bounds. Real libraries have not yet been benchmarked.

## Architecture

| Path | Responsibility |
| --- | --- |
| `crates/proteus-core` | Rust extension `proteus_music._native`: parallel scanning, content-based format detection, tag extraction (PyO3, Lofty, Rayon). |
| `src/proteus_music/models` | Normalized track and organization models. |
| `src/proteus_music/library` | Identity resolution and import service. |
| `src/proteus_music/integrations` | Adapter contract plus local and portable-file adapters. |
| `src/proteus_music/engine` | Features, clustering, assignment, evaluation and naming. Independent of providers and interfaces. |
| `src/proteus_music/playlists` | Organization and export services shared by all interfaces. |
| `src/proteus_music/storage` | SQLite persistence with versioned migrations. |
| `src/proteus_music/cli` | Thin Typer CLI. |

## Contributing

Run the same checks as CI before submitting changes:

```sh
uv run ruff format --check && uv run ruff check && uv run pytest
cargo fmt --all --check && cargo clippy --all-targets -- -D warnings
```

After changing Rust code, `uv sync` rebuilds the extension. Tests use small synthetic fixtures only. Never commit music, databases, credentials or model weights.

## License

Apache-2.0. Third-party data, models, dependencies, and provider content retain their respective licenses and restrictions.
