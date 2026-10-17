"""HTTP API: exact jury endpoints (case §10) plus key-free routes for the web UI."""

import json
import logging
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.jobs import JobStore
from app.media import MediaError, probe
from app.schemas import AnalyzeResponse

log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"
CHUNK = 1024 * 1024

settings = get_settings()
if not settings.api_key:
    raise RuntimeError("API_KEY is not set")
store = JobStore(settings.db_path)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Cash desk video review", docs_url=None, redoc_url=None)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


async def _accept_upload(file: UploadFile) -> AnalyzeResponse:
    job_id = store.new_id()
    # Server-generated name: the client filename is never used as a path.
    dest = settings.uploads_dir / f"{job_id}.mov"
    limit = settings.max_upload_mb * CHUNK
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"file larger than {settings.max_upload_mb} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="empty file")
        await run_in_threadpool(probe, dest)
    except MediaError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"unsupported or corrupt video: {exc}")
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    store.create(job_id, dest)
    log.info("job %s queued (%d bytes)", job_id, size)
    return AnalyzeResponse(job_id=job_id)


def _job_payload(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state == "done":
        return json.loads(Path(job.result_path).read_text(encoding="utf-8"))
    if job.state == "failed":
        return {"job_id": job.id, "status": "failed", "error": job.error}
    return {"job_id": job.id, "status": "processing"}


# --- Jury API (X-API-Key required) ---

@app.post("/api/v1/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_key)])
async def analyze(file: UploadFile) -> AnalyzeResponse:
    return await _accept_upload(file)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str) -> JSONResponse:
    return JSONResponse(_job_payload(job_id))


# --- Web UI (FAQ 27: no user authorization needed) ---

@app.post("/ui/analyze", response_model=AnalyzeResponse)
async def ui_analyze(file: UploadFile) -> AnalyzeResponse:
    return await _accept_upload(file)


@app.get("/ui/jobs/{job_id}")
def ui_get_job(job_id: str) -> JSONResponse:
    return JSONResponse(_job_payload(job_id))


@app.get("/ui/jobs/{job_id}/video")
def ui_video(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # H.264 in a .mov plays in Chrome/Safari when served as mp4.
    return FileResponse(job.input_path, media_type="video/mp4")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
