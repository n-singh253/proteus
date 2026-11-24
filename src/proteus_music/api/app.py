"""FastAPI application for `proteus serve`: thin routes over the CLI's services."""

from __future__ import annotations

import html
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from proteus_music.api.jobs import Job, JobManager
from proteus_music.api.security import LocalOnlyMiddleware, PathRejectedError, validate_path
from proteus_music.api.views import ExportRequest, ImportRequest, JobView, ResultView
from proteus_music.integrations.contract import ExportOptions, ImportBatch
from proteus_music.integrations.files import csv_format, json_format, m3u
from proteus_music.integrations.local import scanner
from proteus_music.library.service import import_batch, library_stats
from proteus_music.models.organization import OrganizationConfig
from proteus_music.playlists import service
from proteus_music.storage import results as result_store
from proteus_music.storage.db import connect
from proteus_music.storage.tracks import load_tracks

STATIC_DIR = Path(__file__).parent / "static"
_SUFFIXES = {"csv": (".csv", ".tsv", ".txt"), "json": (".json",), "m3u": (".m3u", ".m3u8")}
_UNBUILT = """<!doctype html><meta charset="utf-8"><title>Proteus</title>
<p>The web interface has not been built. Run <code>npm --prefix frontend ci</code> and
<code>npm --prefix frontend run build</code>, then restart <code>proteus serve</code>.</p>"""


def create_app(
    *,
    db_path: Path,
    token: str,
    hosts: Iterable[str],
    roots: Iterable[Path],
    static_dir: Path = STATIC_DIR,
) -> FastAPI:
    roots = [Path(r) for r in roots]
    jobs = JobManager()
    app = FastAPI(title="Proteus", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(LocalOnlyMiddleware, token=token, hosts=hosts)
    connect(db_path).close()  # Create or migrate the database before serving.

    @contextmanager
    def database() -> Iterator:
        with closing(connect(db_path)) as conn:
            yield conn

    @app.exception_handler(PathRejectedError)
    async def rejected_path(_, error: PathRejectedError):
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.exception_handler(LookupError)
    async def not_found(_, error: LookupError):
        return JSONResponse({"error": str(error)}, status_code=404)

    @app.get("/api/library")
    def library() -> dict:
        with database() as conn:
            return asdict(library_stats(conn))

    @app.post("/api/imports")
    def start_import(request: ImportRequest) -> JobView:
        if request.kind == "local":
            if request.identification_only:
                raise HTTPException(400, "local folders are your own files; import them normally")
            path = validate_path(request.path, kind="directory", roots=roots)
        else:
            path = validate_path(request.path, kind="file", roots=roots,
                                 suffixes=_SUFFIXES[request.kind])  # fmt: skip

        def work(job: Job) -> dict:
            job.checkpoint("reading files")
            batch = _read(request, path)
            job.checkpoint("saving to library")
            with database() as conn:
                summary = import_batch(conn, batch)
            return {**asdict(summary), "failures": len(summary.failures),
                    "warnings": len(summary.warnings)}  # fmt: skip

        return JobView.of(jobs.submit("import", work))

    @app.post("/api/organize")
    def start_organize(payload: dict) -> JobView:
        try:
            config = OrganizationConfig.model_validate(payload)
        except ValidationError as error:
            messages = [e["msg"].removeprefix("Value error, ") for e in error.errors()]
            raise HTTPException(422, "; ".join(messages)) from None

        def work(job: Job) -> dict:
            with database() as conn:
                result = service.organize_library(conn, config, job.checkpoint)
            return {"result_id": result.id, "playlists": len(result.playlists)}

        return JobView.of(jobs.submit("organize", work))

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JobView:
        return JobView.of(jobs.get(job_id))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> JobView:
        return JobView.of(jobs.cancel(job_id))

    @app.get("/api/results")
    def results() -> list[dict]:
        with database() as conn:
            rows = result_store.list_results(conn)
        return [{"id": r, "created_at": c.isoformat(), "playlists": p, "tracks": t}
                for r, c, p, t in rows]  # fmt: skip

    @app.get("/api/results/{result_id}")
    def result(result_id: str) -> ResultView:
        with database() as conn:
            loaded = service.get_result(conn, result_id)  # Validated before it is shown.
            return ResultView.of(loaded, {t.id: t for t in load_tracks(conn)})

    @app.post("/api/results/{result_id}/export")
    def start_export(result_id: str, request: ExportRequest) -> JobView:
        out_dir = validate_path(request.out_dir, kind="output", roots=roots)
        options = ExportOptions(relative_paths=request.relative_paths, overwrite=request.overwrite)

        def work(job: Job) -> dict:
            job.checkpoint("writing files")
            with database() as conn:
                report = service.export_result(
                    conn, request.target, out_dir, result_id=result_id, options=options
                )
            return {"files": list(report.files), "exported_tracks": report.exported_tracks,
                    "unexportable": [i.reason for i in report.unexportable]}  # fmt: skip

        return JobView.of(jobs.submit("export", work))

    index = static_dir / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        if not index.is_file():
            return _UNBUILT
        return index.read_text(encoding="utf-8").replace("__PROTEUS_TOKEN__", html.escape(token))

    if (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
    return app


def _read(request: ImportRequest, path: Path) -> ImportBatch:
    permitted = not request.identification_only
    if request.kind == "local":
        return scanner.read_directory(path, recursive=request.recursive)
    if request.kind == "csv":
        return csv_format.read_csv(path, separator=request.separator, analysis_permitted=permitted)
    if request.kind == "json":
        return json_format.read_json(path, analysis_permitted=permitted)
    return m3u.read_m3u(path, analysis_permitted=permitted)
