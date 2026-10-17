"""Single persistent worker: claims queued jobs from SQLite and runs the pipeline.

Run as its own process (python -m app.worker) so the API stays responsive during inference.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from app import pipeline
from app.config import Settings, get_settings
from app.jobs import Job, JobStore
from app.media import MediaError, decode
from app.rules import RuleRegistry
from app.vlm import VLMError, build_vlm

log = logging.getLogger("worker")


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def process(job: Job, store: JobStore, vlm, reg: RuleRegistry, s: Settings) -> None:
    started = time.monotonic()
    # The jury clock starts at upload, so queue wait counts against the budget.
    deadline = started + max(30.0, s.job_budget_sec - (time.time() - job.created_at))
    try:
        video = decode(Path(job.input_path), s.verify_fps, s.model_max_side)
        decode_sec = time.monotonic() - started
        events, stats = asyncio.run(pipeline.run(video, vlm, reg, s, deadline))
        result = pipeline.build_result(job.id, events, video, reg)
    except (MediaError, VLMError) as exc:
        log.warning("job %s failed: %s", job.id, exc)
        store.fail(job, str(exc))
        return
    except Exception as exc:
        log.exception("job %s crashed", job.id)
        store.fail(job, f"internal error: {type(exc).__name__}")
        return

    out = s.results_dir / f"{job.id}.json"
    write_atomic(out, result.model_dump())
    # Timing and counts only: no model text goes to the log.
    meta = {
        "job_id": job.id, "duration_sec": round(video.info.duration_sec, 2), "frames": len(video.frames),
        "decode_sec": round(decode_sec, 2), "total_sec": round(time.monotonic() - started, 2),
        "queue_sec": round((job.started_at or job.created_at) - job.created_at, 2),
        "verdict": result.verdict, "events": len(result.violations), "rules_version": reg.version, **stats,
    }
    write_atomic(s.results_dir / f"{job.id}.meta.json", meta)
    if not store.finish(job, out):
        log.warning("job %s: stale claim, result discarded", job.id)
    log.info("job %s done in %.1fs", job.id, meta["total_sec"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    s = get_settings()
    store = JobStore(s.db_path)
    reg = RuleRegistry.load(s.rules_path)
    vlm = build_vlm(s)
    requeued = store.recover_interrupted()
    if requeued:
        log.info("requeued %d interrupted job(s)", requeued)
    log.info("worker ready: backend=%s model=%s rules=%s", s.vlm_backend, s.vlm_model, reg.version)
    while True:
        job = store.claim_next()
        if job is None:
            time.sleep(s.worker_poll_sec)
            continue
        process(job, store, vlm, reg, s)


if __name__ == "__main__":
    main()
