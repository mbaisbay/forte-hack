"""Persistent job queue in SQLite. The API inserts, one worker process claims."""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    state        TEXT NOT NULL,          -- queued | running | done | failed
    input_path   TEXT NOT NULL,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    attempt      INTEGER NOT NULL DEFAULT 0,
    claim_token  TEXT,
    result_path  TEXT,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state_created ON jobs(state, created_at);
"""

MAX_ATTEMPTS = 2


@dataclass
class Job:
    id: str
    state: str
    input_path: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    attempt: int
    claim_token: str | None
    result_path: str | None
    error: str | None


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def create(self, job_id: str, input_path: Path) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs (id, state, input_path, created_at) VALUES (?, 'queued', ?, ?)",
                (job_id, str(input_path), time.time()),
            )

    def get(self, job_id: str) -> Job | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job(**dict(row)) if row else None

    def claim_next(self) -> Job | None:
        """Atomically move the oldest queued job to running and return it."""
        token = uuid.uuid4().hex
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT id FROM jobs WHERE state = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                c.execute("COMMIT")
                return None
            c.execute(
                "UPDATE jobs SET state = 'running', started_at = ?, attempt = attempt + 1, claim_token = ? "
                "WHERE id = ?",
                (time.time(), token, row["id"]),
            )
            c.execute("COMMIT")
        return self.get(row["id"])

    def finish(self, job: Job, result_path: Path) -> bool:
        """Mark done only if this worker still holds the claim. Returns False on a stale write."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET state = 'done', finished_at = ?, result_path = ?, error = NULL "
                "WHERE id = ? AND claim_token = ? AND state = 'running'",
                (time.time(), str(result_path), job.id, job.claim_token),
            )
        return cur.rowcount == 1

    def fail(self, job: Job, error: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET state = 'failed', finished_at = ?, error = ? "
                "WHERE id = ? AND claim_token = ? AND state = 'running'",
                (time.time(), error[:500], job.id, job.claim_token),
            )
        return cur.rowcount == 1

    def recover_interrupted(self) -> int:
        """Called at worker start: requeue jobs left running by a crash, fail repeat offenders."""
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET state = 'failed', finished_at = ?, error = 'interrupted twice' "
                "WHERE state = 'running' AND attempt >= ?",
                (time.time(), MAX_ATTEMPTS),
            )
            cur = c.execute(
                "UPDATE jobs SET state = 'queued', claim_token = NULL WHERE state = 'running'"
            )
        return cur.rowcount
