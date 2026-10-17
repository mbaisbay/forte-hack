# ForteBank Datathon 26 — build plan

Goal: a public URL where the jury uploads a `.mov` and, within 5 minutes, gets the exact JSON
(verdict + typed violations + timestamps + 1–3 JPEG evidence frames + regulation reference),
with all inference running locally on the RTX 5090.

Data rule (kept from the brief): the cloud coding agent never sees video content, frames,
or model descriptions of them. On the 5090, scripts print **only aggregate numbers** to the agent.
Failure inspection is done by a teammate locally.

---

## Hardware and model

| Item | Choice | Why |
|---|---|---|
| GPU | RTX 5090, 32 GB VRAM (Blackwell, sm_120) | Team's machine, via ssh |
| Model | `Qwen/Qwen3-VL-8B-Instruct`, bf16 | ~17 GB weights, leaves ~12 GB for KV cache; Apache-2.0 |
| Fallback (speed) | `Qwen/Qwen3-VL-4B-Instruct` | Same API, half the size |
| Upgrade (only if 8B measured weak) | Qwen3-VL-30B-A3B in 4-bit (AWQ) | MoE, fits ~18–20 GB; test only after baseline works |
| Serving | vLLM, OpenAI-compatible server bound to `127.0.0.1` | Fast prefill, batches windows of one video in parallel |
| Requirements | NVIDIA driver ≥ 570, CUDA 12.8+, PyTorch with sm_120 support | Blackwell needs these; older wheels fail with "no kernel image" |

Do **not** use Ultralytics YOLO: AGPL-3.0, not commercially permissive (FAQ 24). If a detector is
ever needed, use an Apache/MIT one (RT-DETR via transformers, YOLOX).

---

## Pipeline

```
upload .mov ──► validate (PyAV decode 1 frame) ──► SQLite job (queued) ──► job_id returned
                                                      │
                                     worker process (separate from API)
                                                      │
        decode with real timestamps ──► sample 2 fps ──► 10 s windows, 2 s overlap
                                                      │
        Pass 1, per window (parallel in vLLM): frames labeled "F012 t=6.0s"
          → JSON: observed actions + candidate rule_ids + frame_ids + confidence
                                                      │
        merge candidates across windows (same rule_id, overlapping time)
                                                      │
        Pass 2, per candidate: denser frames (4–6 fps, ±3 s) → supported / not_supported / unclear
          + best evidence frame_ids
                                                      │
        verdict rule: any supported candidate ≥ threshold → "violation"
        (threshold tuned on dev clips, biased to recall: misses are "unacceptable", FAQ 16)
                                                      │
        attach regulation text from rules.json (never from the model)
        resolve frame_ids → original-frame JPEGs → base64
                                                      │
        validate against jury schema ──► result.json ──► job "done"
```

Unknown categories (FAQ 9): catalogue includes an `other_suspicious` entry. It counts for the
verdict (25 pts) but gets a generic, honest description and the closest general regulation.

Latency estimate (not measured): 142 s clip × 2 fps ≈ 285 frames ≈ 100k image tokens before
downscaling. On a 5090 this should be well under 60 s. Measure in Phase C; if over, downscale frames
or drop to 1.5 fps.

---

## Repo layout

```
app/
  main.py        # FastAPI: /api/v1/analyze, /api/v1/jobs/{id}, UI routes, X-API-Key
  schemas.py     # jury response models
  jobs.py        # SQLite job table
  worker.py      # claims queued jobs, runs pipeline, writes result
  media.py       # PyAV decode, timestamps, frame sampling, JPEG/base64
  vlm.py         # OpenAI-compatible client, refuses non-localhost base URL
  pipeline.py    # windows → pass 1 → merge → pass 2 → verdict
  rules.py       # load + validate rules.json
  static/        # index.html + app.js (no framework, no CDN)
config/rules.json
prompts/observe.txt, verify.txt
scripts/
  setup_5090.sh      # driver check, venv, vLLM, weight download
  serve_vlm.sh       # vLLM on 127.0.0.1:8000
  run.sh             # API + worker
  eval_dev.py        # runs all 42 dev clips, prints ONLY aggregate metrics
  make_synthetic.py  # synthetic test video (no competition data)
tests/               # contract + failure tests with synthetic video
```

`.gitignore`: `open/`, `data/`, `jobs/`, `models/`, `.env`.

---

## Phases

### A. On this Mac (agent builds; no GPU needed) — about 3–4 h
1. Repo skeleton, `.gitignore`, config — 15 min
2. API contract + `X-API-Key` + contract tests — 40 min
3. SQLite jobs + separate worker process — 40 min
4. Media: decode, timestamps, sampling, JPEG evidence — 40 min
5. VLM client + mock adapter (clearly marked TEST ONLY) — 20 min
6. Pipeline: windows, merge, verify, verdict, schema validation — 1 h
7. Single-page UI (Russian): upload, status, verdict, event cards, frames, click-to-seek — 45 min
8. `eval_dev.py` (aggregate-only output) — 20 min

### B. On the 5090 (can start now, in parallel with A) — about 1 h, mostly downloads
1. `nvidia-smi`: confirm driver ≥ 570 and CUDA 12.8+
2. Install Python 3.12 + uv, vLLM with Blackwell support, ffmpeg
3. Download Qwen3-VL-8B-Instruct (~17 GB)
4. Start vLLM on `127.0.0.1:8000`, smoke-test with a synthetic image
5. Public URL: tunnel (see open decisions) → verify from phone on mobile data

### C. Integrate and tune on the 5090 — about 3–4 h
1. Copy repo to the PC (rsync over ssh), copy `open/` to the PC
2. First real end-to-end run: one dev clip → full JSON  ← **gate: must work by hour 5**
3. Run `eval_dev.py` on all 42: video-level accuracy, misses, false alarms, timestamp error
4. Teammate inspects failures locally → adjusts prompts and catalogue
5. Measure the longest clip (Х-14, 142.5 s) end-to-end; must be ≤ 240 s

### D. Regulations (parallel, starts now) — about 2 h
1. Agent searches public KZ sources (НБРК, adilet.zan.kz) for cash operations, collection (инкассация),
   cash-room security. No competition data involved.
2. When the teammate's violation-type list is ready → map each type to a source + section + quote
3. Human reviews each mapping → `rules.json`, `verified: true`

### E. Submission — about 1.5 h
1. Offline test: block outbound internet on the PC, process a clip
2. Restart test, corrupt-file test, wrong-key test
3. README (launch steps, architecture, models + licenses), types list, sources, limitations
4. Final check from an outside network; send URL + API key to the jury
