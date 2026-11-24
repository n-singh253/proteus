"""Local web API: protections, the full workflow, and no partial results on failure."""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from proteus_music.api.app import create_app
from proteus_music.api.jobs import JobManager, JobState
from proteus_music.cli.main import app as cli
from proteus_music.models.organization import OrganizationConfig
from proteus_music.playlists.service import organize_library
from proteus_music.storage.db import connect
from proteus_music.storage.results import list_results

TOKEN = "test-token"
HEADERS = {"X-Proteus-Token": TOKEN}


@pytest.fixture
def setup(tmp_path):
    lines = ["title,artist,genres,tags"]
    for genre, tags in (("Hip-Hop", "rap;energetic"), ("Classical", "calm;strings")):
        lines += [f"{genre} {i},{genre} Artist {i % 3},{genre},{tags}" for i in range(20)]
    lines.append("Untagged,,,")
    (tmp_path / "library.csv").write_text("\n".join(lines) + "\n")
    db = tmp_path / "lib.db"
    app = create_app(db_path=db, token=TOKEN, hosts={"testserver"}, roots=[tmp_path])
    return TestClient(app), db, tmp_path


def post(client, path, body, **headers):
    return client.post(path, json=body, headers={**HEADERS, **headers})


def run_job(client, path, body):
    job = post(client, path, body).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}", headers=HEADERS).json()
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def import_csv(client, root):
    return run_job(client, "/api/imports", {"kind": "csv", "path": str(root / "library.csv")})


def test_requests_must_be_local_and_carry_the_session_token(setup):
    client, _, _ = setup
    assert client.get("/api/library").status_code == 403
    assert client.get("/api/library", headers={"X-Proteus-Token": "guess"}).status_code == 403
    rebinding = client.get("/api/library", headers={**HEADERS, "Host": "evil.example:8765"})
    assert rebinding.status_code == 400
    cross = post(client, "/api/organize", {"mode": "auto"}, Origin="https://evil.example")
    assert cross.status_code == 403
    assert client.get("/api/library", headers=HEADERS).status_code == 200


def test_user_paths_are_validated(setup):
    client, _, root = setup
    cases = (
        ("library.csv", "absolute"),
        ("/etc/hosts", "outside the allowed"),
        (str(root / "missing.csv"), "no file"),
        (str(root / "lib.db"), "expected a"),
    )
    for path, message in cases:
        response = post(client, "/api/imports", {"kind": "csv", "path": path})
        assert response.status_code == 400 and message in response.json()["error"]


def test_ui_flow_shares_results_with_the_cli(setup):
    client, db, root = setup
    imported = import_csv(client, root)
    assert imported["state"] == "succeeded" and imported["result"]["new_tracks"] == 41

    organized = run_job(client, "/api/organize", {"mode": "auto"})
    assert organized["state"] == "succeeded"
    result_id = organized["result"]["result_id"]
    view = client.get(f"/api/results/{result_id}", headers=HEADERS).json()
    names = {p["name"] for p in view["playlists"]}
    assert names == {"Hip-Hop", "Classical", "Unclassified"}
    flagged = [t["title"] for p in view["playlists"] for t in p["flagged_tracks"]]
    assert flagged == ["Untagged"]

    report = CliRunner().invoke(cli, ["--db", str(db), "report", result_id, "--json"])
    assert {p["name"] for p in json.loads(report.output)["playlists"]} == names

    out = root / "exports"
    body = {"target": "csv", "out_dir": str(out)}
    exported = run_job(client, f"/api/results/{result_id}/export", body)
    assert exported["state"] == "succeeded" and (out / "playlists.csv").is_file()


def test_failed_or_cancelled_jobs_leave_no_partial_result(setup):
    client, db, root = setup
    import_csv(client, root)
    infeasible = run_job(client, "/api/organize", {"playlists": 100})
    assert infeasible["state"] == "failed" and "need 100 tracks" in infeasible["reasons"][0]
    assert post(client, "/api/organize", {"playlists": 2, "min_playlists": 1}).status_code == 422

    def cancel_at(stage):
        def checkpoint(current):
            if current == stage:
                raise RuntimeError("cancelled")

        return checkpoint

    config = OrganizationConfig(playlists=2)
    with connect(db) as conn:
        for stage in ("organizing", "saving"):  # "saving" is the last point before the write.
            with pytest.raises(RuntimeError):
                organize_library(conn, config, cancel_at(stage))
        assert list_results(conn) == []

    jobs, release = JobManager(), threading.Event()
    blocker = jobs.submit("block", lambda job: release.wait() and {})
    queued = jobs.submit(
        "organize", lambda job: organize_library(connect(db), config, job.checkpoint)
    )
    jobs.cancel(queued.id)
    release.set()
    for _ in range(200):
        if queued.state is not JobState.QUEUED and blocker.state is JobState.SUCCEEDED:
            break
        time.sleep(0.01)
    assert queued.state is JobState.CANCELLED
    with connect(db) as conn:
        assert list_results(conn) == []
