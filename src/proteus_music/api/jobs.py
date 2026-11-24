"""One-at-a-time background jobs with progress and cooperative cancellation."""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from proteus_music.engine.optimization.assignment import InfeasibleConstraintsError

log = logging.getLogger(__name__)
MAX_KEPT_JOBS = 100


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCancelledError(Exception):
    """Raised at a checkpoint after cancellation was requested."""


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = JobState.QUEUED
    stage: str = "queued"
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    reasons: list[str] = field(default_factory=list)
    cancel_requested: threading.Event = field(default_factory=threading.Event)

    def checkpoint(self, stage: str, progress: float | None = None) -> None:
        """Report progress; stop here if the user cancelled. Work done so far must be
        uncommitted or already complete, so stopping leaves no partial state."""
        if self.cancel_requested.is_set():
            raise JobCancelledError
        self.stage, self.progress = stage, progress


Work = Callable[[Job], dict[str, Any]]


class JobManager:
    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: queue.Queue[tuple[Job, Work]] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._run, name="proteus-jobs", daemon=True).start()

    def submit(self, kind: str, work: Work) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_KEPT_JOBS:
                self._jobs.popitem(last=False)
        self._queue.put((job, work))
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise LookupError(f"no job {job_id!r}")
            return self._jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state in (JobState.QUEUED, JobState.RUNNING):
            job.cancel_requested.set()
        return job

    def _run(self) -> None:
        while True:
            job, work = self._queue.get()
            self._execute(job, work)

    @staticmethod
    def _execute(job: Job, work: Work) -> None:
        if job.cancel_requested.is_set():
            job.state, job.stage = JobState.CANCELLED, "cancelled before start"
            return
        job.state = JobState.RUNNING
        try:
            job.result = work(job)
            job.state, job.stage, job.progress = JobState.SUCCEEDED, "done", 1.0
        except JobCancelledError:
            job.state, job.stage = JobState.CANCELLED, f"cancelled during {job.stage}"
        except InfeasibleConstraintsError as error:
            job.state, job.error, job.reasons = JobState.FAILED, "infeasible", error.reasons
        except ValidationError as error:
            job.state = JobState.FAILED
            job.error = "; ".join(e["msg"].removeprefix("Value error, ") for e in error.errors())
        except (LookupError, ValueError, OSError) as error:
            job.state, job.error = JobState.FAILED, str(error)
        except Exception as error:  # Report unexpected failures instead of losing the job.
            log.exception("job %s failed", job.id)
            job.state, job.error = JobState.FAILED, f"unexpected error: {error}"
