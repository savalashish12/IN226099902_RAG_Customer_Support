"""
worker.py — Background ingestion worker with thread-safe progress tracking.

Runs batch PDF ingestion in a daemon thread so the Streamlit UI never
blocks. Progress is stored in a module-level, lock-protected registry that
the UI polls on a timer to render a live progress bar and per-file status.

Since Streamlit reruns the script on every interaction, the worker state
lives at module scope (not in session_state) keyed by a job_id.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import threading
from datetime import datetime

from ingestion.batch import ingest_one


class IngestJob:
    """A single background batch-ingestion job with live progress."""

    def __init__(self, job_id: str, items: list[tuple[str, bytes]]):
        self.job_id     = job_id
        self.items      = items
        self.total      = len(items)
        self.completed  = 0
        self.results    : list[dict] = []
        self.status     = "running"      # running | done | error
        self.error      = ""
        self.started_at = datetime.now().isoformat()
        self.finished_at: str | None = None
        self._lock      = threading.Lock()
        self._thread    : threading.Thread | None = None

    def _run(self):
        try:
            for name, data in self.items:
                res = ingest_one(data, name)
                with self._lock:
                    self.results.append({
                        "name": res.name, "status": res.status,
                        "chunks": res.chunks, "collection": res.collection,
                        "error": res.error,
                    })
                    self.completed += 1
            with self._lock:
                self.status = "done"
                self.finished_at = datetime.now().isoformat()
        except Exception as e:
            with self._lock:
                self.status = "error"
                self.error = str(e)
                self.finished_at = datetime.now().isoformat()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id"    : self.job_id,
                "total"     : self.total,
                "completed" : self.completed,
                "results"   : list(self.results),
                "status"    : self.status,
                "error"     : self.error,
                "progress"  : (self.completed / self.total) if self.total else 1.0,
                "finished_at": self.finished_at,
            }

    def first_collection(self) -> str | None:
        with self._lock:
            for r in self.results:
                if r.get("collection"):
                    return r["collection"]
        return None


# Registry of active/recent jobs, keyed by job_id
_jobs: dict[str, IngestJob] = {}
_jobs_lock = threading.Lock()


def start_job(job_id: str, items: list[tuple[str, bytes]]) -> IngestJob:
    """Create and start a background ingestion job."""
    job = IngestJob(job_id, items)
    with _jobs_lock:
        _jobs[job_id] = job
    job.start()
    return job


def get_job(job_id: str) -> IngestJob | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def prune_jobs(keep: int = 5):
    """Keep only the most recent `keep` jobs to bound memory."""
    with _jobs_lock:
        if len(_jobs) <= keep:
            return
        ordered = sorted(_jobs.values(), key=lambda j: j.started_at)
        for job in ordered[:-keep]:
            _jobs.pop(job.job_id, None)
