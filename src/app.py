"""Service: web UI + jury API (case section 10).
  set API_KEY=... ; .venv/Scripts/python -m uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
POST /api/v1/analyze (file=.mov, header X-API-Key) -> {"job_id"}
GET  /api/v1/jobs/{job_id}                         -> {"job_id","status","verdict","violations":[...]}
"""
import os, queue, threading, traceback, uuid
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "uploads"; UP.mkdir(exist_ok=True)
API_KEY = os.environ.get("API_KEY", "change-me")
JOBS: dict = {}
Q: "queue.Queue[str]" = queue.Queue()
app = FastAPI(title="ForteBank cash-desk violation detector")


def worker():
    import detect  # loads torch lazily in the worker thread
    while True:
        jid = Q.get()
        try:
            res = detect.analyze(JOBS[jid]["path"])
            res.pop("debug", None)
            JOBS[jid].update(status="done", **res)
        except Exception:
            JOBS[jid].update(status="failed", error=traceback.format_exc()[-500:])


threading.Thread(target=worker, daemon=True).start()


def check(key):
    if key != API_KEY:
        raise HTTPException(401, "invalid API key")


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check(x_api_key)
    jid = uuid.uuid4().hex[:8]
    path = UP / f"{jid}{Path(file.filename or 'v.mov').suffix or '.mov'}"
    path.write_bytes(await file.read())
    JOBS[jid] = {"job_id": jid, "status": "processing", "path": str(path)}
    Q.put(jid)
    return {"job_id": jid}


@app.get("/api/v1/jobs/{jid}")
def job(jid: str, x_api_key: str = Header(None)):
    check(x_api_key)
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in JOBS[jid].items() if k != "path"}


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "src/ui.html").read_text(encoding="utf-8")
