# Cash-desk violation detector: service

A local AI service for ForteBank Datathon 26. It analyses cashier CCTV videos and shows possible violations with a timecode, evidence frames and the rule source. Everything runs locally on one GPU (12 GB). No cloud AI is used.

## Pages
| Page | URL | What it does |
|---|---|---|
| Videos | `/` | Upload a video ("analysis in progress, please wait" until done). Every video is stored, with its verdict, violations, 3 evidence frames and a player that jumps to each timecode. A reviewer labels each violation (correct / wrong type or time / false alarm), sets the overall verdict and adds missed violations. Labels are saved. |
| Rules | `/rules` | The rule catalogue (id, source, norm text, what the camera sees), editable by hand. Upload a regulatory document (.pdf/.docx/.txt) and the local LLM proposes rules; a person edits and accepts or rejects them. Also shows the extraction-quality test on our catalogue. |
| Decision tree | `/tree` | The checks the verdict is made from. Turn checks on or off, change thresholds, rule mapping and the reviewer text. LLM suggestions link newly accepted rules to existing checks. Rules with no automatic check are listed. |

## API
Jury API (`X-API-Key` header, as in case section 10):
```
POST /api/v1/analyze   file=@video.mov            -> {"job_id": "..."}
GET  /api/v1/jobs/{job_id}                        -> {"job_id","status": processing|done|failed,"verdict","violations":[...]}
```
Each violation has `type, timecode_sec, description, evidence_frames[1-3]{timestamp_sec, image_base64}`, plus `rule_id, source, check`.

UI API (no key, used by the pages):
- `/api/ui/videos`: upload and list videos
- `/api/ui/videos/{id}`, `.../file`, `.../label` (PUT), `.../rerun` (POST)
- `/api/ui/labels`: all labels next to the model output, for re-tuning
- `/api/ui/rules` (GET/PUT), `/api/ui/tree` (GET/PUT)
- `/api/ui/docs`, `/api/ui/docs/{id}/suggestions/{i}` (POST accept/reject), `/api/ui/mapping_suggestions`, `/api/ui/eval_rules`

## Setup (Python 3.12, NVIDIA GPU >= 12 GB, CUDA 12.8 driver)
```
python -m venv .venv && .venv/Scripts/activate      # Linux: source .venv/bin/activate
pip install -r requirements.txt
hf auth login                                       # needed for gated google/gemma-3-4b-it
python download_models.py                           # -> models/ (~33 GB; Qwen3-8B only for document parsing)
```
Set `MODELS_DIR` if the weights are stored elsewhere. After the download the service needs no internet.

## Run
```
API_KEY=<secret> python -m uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
```
To publish it, put the port behind a public URL (e.g. a reverse proxy or tunnel).

## How it works
1. **Frames.** About 1 frame per second over the whole video, at most 60 frames.
2. **Models.** They load one at a time to fit 12 GB:
   - YOLO11m (CPU): people and the laptop
   - Qwen3-VL-4B: locates the safe, door and desk; desk yes/no questions; safe open/closed
   - Gemma-3-4B: desk questions, used for personal items and cash
3. **Pixel checks.** Laptop screen brightness, end vs start (locked or not); scene change outside people and the laptop (door).
4. **Decision tree** (`data/tree.json`, edited on `/tree`) turns the signals into timed violations. Rules and sources come from `data/rules.json`, edited on `/rules`.
5. **Single GPU queue.** Videos take priority over document parsing, so a document never delays a jury video by more than one paragraph (~4 s).

Storage: `storage/service.db` (SQLite) holds videos, results, labels, documents and suggestions. Uploaded files are in `storage/videos` and `storage/docs`.

## Measured quality (40 training videos)
- **Verdict.** On the training angle: recall 24/24, precision 0.80, F2 0.952. On a simulated new angle (flip + rotation + crop): recall 23/24, precision 0.79, F2 0.920.
- **Speed.** About 53 s per video on average, at most about 100 s. Real-time factor about 1.6 (1 s of video takes about 1.6 s).
- **Rule extraction on our catalogue** (Qwen3-8B, 90 s):
  - 24/24 paragraphs extracted
  - 10/12 of our rules marked visible on camera
  - 4/5 correct check mapping
  - sources are built by code from the document structure, so no invented references

## Known limits
- Violations about the banknotes themselves (mixed denominations, sorting) are not visible to 4B models at this resolution.
- **Missing objects.** If the safe or laptop is not found, that check silently does not fire.
- **Camera-specific threshold.** `person_min_y` (people behind the glass wall) depends on the camera. Check it on `/tree` for a new angle.
- **Recall over precision.** Clean videos with a client at the desk are often flagged (the "client_fallback" check), on purpose.
- **Rule 53.26.1** (laptop left unlocked) cites a placeholder source. Confirm it before submission.
- **Timecodes** are estimates (±2 s for about two thirds of the events).
