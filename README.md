# Cash desk video review (ForteBank Datathon 26)

Local AI service that reviews overhead video of a cashier workstation and flags visible signs of
possible cash-discipline violations, with timestamps, original evidence frames, and the regulation
each finding relates to. All inference runs on the team's GPU. No cloud AI is called.

The result is a hint for a human reviewer, not a legal or disciplinary conclusion.

## Architecture

```
Browser UI ─┐                       ┌─ SQLite job queue ─ worker process ─┐
Jury client ─┴─ FastAPI (port 8080) ┘                                     │
                                         PyAV decode (real timestamps, 6 fps grid)
                                         Pass 1: 10 s windows @ 2 fps ──► local VLM (vLLM, 127.0.0.1:8000)
                                         merge duplicates across windows
                                         Pass 2: denser re-check of each candidate ──► local VLM
                                         regulation text from config/rules.json (never from the model)
                                         evidence = original-resolution JPEG frames
```

- Model: `Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0), served by vLLM on loopback only.
- The VLM client refuses any non-loopback URL (`app/vlm.py`).
- Any model failure marks the job `failed`; it never becomes a silent `no_violation`.

## API (for the jury)

```bash
curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: <key>" -F "file=@video.mov"
# -> {"job_id": "..."}
curl https://<host>/api/v1/jobs/<job_id> -H "X-API-Key: <key>"
# -> {"job_id","status":"processing"} | {"job_id","status":"failed","error"} |
#    {"job_id","status":"done","verdict":"violation"|"no_violation","violations":[...]}
```

Each violation: `type`, `timecode_sec`, `description` (what is visible, why it may be a violation,
regulation reference), `evidence_frames` (1–3 JPEG, base64).

## Run on the GPU machine

```bash
./scripts/setup_5090.sh                 # once, needs internet: venvs, vLLM, model weights
cp .env.example .env && nano .env       # set API_KEY
./scripts/serve_vlm.sh                  # terminal 1: model server (offline from here on)
./scripts/run.sh                        # terminal 2: API + worker on 127.0.0.1:8080
cloudflared tunnel --url http://127.0.0.1:8080   # terminal 3: public URL
```

Shared GPU: `GPU_UTIL` (default 0.85) is the share of total VRAM vLLM takes and must be free at start.

## Develop and test (no GPU)

```bash
uv sync
uv run pytest          # synthetic video only, mock model
```

## Evaluate on dev clips

```bash
.venv/bin/python scripts/eval_dev.py --subset dev        # prints aggregate metrics only
.venv/bin/python scripts/eval_dev.py --subset holdout    # use sparingly, after changes settle
```

Per-clip details go to `eval_details.json` (git-ignored) for local inspection.

## Data handling

Competition videos, frames, model outputs and per-clip results stay on the team's machines and are
not committed (`.gitignore`). Delete `data/` and `open/` after the hackathon.
