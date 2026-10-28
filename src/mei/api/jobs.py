"""Lightweight, thread-safe asynchronous job manager with persistent state.

Stores job history as JSON files in artifacts/jobs/<job_id>.json so server
restarts never lose job records.
"""

import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mei.api.models import JobStatusResponse
from mei.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


class JobManager:
    """Thread pool executor and persistent state store for background jobs."""

    _instance: Optional["JobManager"] = None
    _lock = threading.Lock()

    def __init__(self, jobs_dir: Optional[Path] = None, max_workers: int = 2) -> None:
        self.jobs_dir = Path(jobs_dir) if jobs_dir else ARTIFACTS_DIR / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mei_job"
        )
        self._memory_jobs: Dict[str, Dict[str, Any]] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._load_persisted_jobs()

    @classmethod
    def get_instance(cls) -> "JobManager":
        """Get or create singleton job manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = JobManager()
            return cls._instance

    def _load_persisted_jobs(self) -> None:
        """Load past jobs from disk into memory."""
        for p in self.jobs_dir.glob("job_*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    job_data = json.load(f)
                jid = job_data.get("job_id")
                if jid:
                    self._memory_jobs[jid] = job_data
            except Exception as e:
                logger.debug("Failed loading persisted job %s: %s", p.name, e)

    def _persist_job(self, job_data: Dict[str, Any]) -> None:
        """Atomically persist job state to JSON file."""
        jid = job_data["job_id"]
        job_file = self.jobs_dir / f"{jid}.json"
        try:
            with open(job_file, "w", encoding="utf-8") as f:
                json.dump(job_data, f, indent=2)
        except Exception as e:
            logger.error("Failed persisting job %s: %s", jid, e)

    def submit_job(
        self,
        job_type: str,
        task_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Create and submit a background task."""
        now = datetime.now(timezone.utc).isoformat()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        cancel_event = threading.Event()
        self._cancel_flags[job_id] = cancel_event

        job_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "progress": 0.0,
            "message": "Task queued for background execution",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }

        with self._lock:
            self._memory_jobs[job_id] = job_data
            self._persist_job(job_data)

        def runner() -> None:
            self.update_job(
                job_id, status="running", progress=0.05, message="Task running"
            )
            try:
                # Inject update_fn and cancel_event into task if accepted
                result = task_fn(
                    *args,
                    update_fn=lambda p, msg: self.update_job(
                        job_id, progress=p, message=msg
                    ),
                    cancel_event=cancel_event,
                    **kwargs,
                )
                if cancel_event.is_set():
                    self.update_job(
                        job_id,
                        status="cancelled",
                        progress=1.0,
                        message="Task cancelled by user",
                    )
                else:
                    self.update_job(
                        job_id,
                        status="finished",
                        progress=1.0,
                        message="Task completed successfully",
                        result=result,
                    )
            except Exception as exc:
                logger.exception("Error executing background job %s: %s", job_id, exc)
                self.update_job(
                    job_id,
                    status="failed",
                    progress=1.0,
                    message=str(exc),
                    error=str(exc),
                )

        self.executor.submit(runner)
        return job_id

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update job progress, message, and status."""
        with self._lock:
            if job_id not in self._memory_jobs:
                return
            job = self._memory_jobs[job_id]
            now = datetime.now(timezone.utc).isoformat()
            if status is not None:
                job["status"] = status
            if progress is not None:
                job["progress"] = max(0.0, min(1.0, float(progress)))
            if message is not None:
                job["message"] = message
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error
            job["updated_at"] = now
            self._persist_job(job)

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        """Fetch job details by ID."""
        with self._lock:
            data = self._memory_jobs.get(job_id)
            if data:
                return JobStatusResponse(**data)
        return None

    def list_jobs(self) -> List[JobStatusResponse]:
        """List all background jobs sorted by created_at descending."""
        with self._lock:
            jobs = [JobStatusResponse(**d) for d in self._memory_jobs.values()]
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Signal job cancellation."""
        with self._lock:
            event = self._cancel_flags.get(job_id)
            if event:
                event.set()
                self.update_job(
                    job_id, status="cancelled", message="Cancellation requested"
                )
                return True
        return False
