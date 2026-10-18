# Final results: cashier CCTV violation detector

Setup: 40 unique videos (Х-47 = Х-4 and Х-48 = Х-5 are merged), 24 with a violation and 16 clean. The single GPU is an RTX 5070 with 12 GB. Only pretrained models are used, with no fine-tuning.

Metric priority: recall first, then F2 (recall weighted above precision). The baseline that always answers "violation" scores F2 0.882 (precision 0.60, recall 1.00).

| # | Hypothesis | Recall | Precision | F2 | Time per video |
|---|---|---|---|---|---|
| 1 | Feature ensemble + OR-policy v7 (precomputed features, leave-one-out thresholds) | 24/24 | 24/29 = 0.83 | **0.960** | same as #2 |
| 2 | The same policy run end to end on raw video (`src/detect.py`) | 24/24 | 24/30 = 0.80 | 0.952 | 40–90 s, max 89 s |
| 2b | #2 on the simulated new camera angle (flip + 4° rotation + crop) | 23/24 | 23/29 = 0.79 | 0.920 | max 99 s |
| 3 | Classic ML on the model features (greedy OR "Cover", LR, RF, ET; leave-one-out) | lower than #1 (exact scores not saved here) | | | features as #2, training under 1 s |

Rejected: a single-prompt few-shot VLM, a rolling narration plus a text judge, the 8B/12B models, a mixed-denominations check, and Gemma reading the laptop or safe. See the end of this file and `runs/journal.md`.

---

## H1 (best): a feature ensemble with one simple model per signal, combined by an OR-policy

Idea: no model judges the whole scene. Each model answers one narrow, visible question per frame. A transparent rule set (a state machine over start and end states) turns those answers into violations.

### Pipeline (`src/detect.py::analyze`)
1. **Sample frames.** `frames.sample(path, fps=1, max_frames=60, width=464)` takes evenly spaced frames over the whole video. Videos longer than 60 s get a wider spacing. cv2 can't open Cyrillic paths, so `frames._open` copies the file to a temporary ASCII path first.
2. **YOLO11m (COCO, CPU)** detects classes 0 (person) and 63 (laptop) at conf 0.25.
   - A person counts only with conf ≥ 0.35 and a box bottom y2 > 0.40 (normalized). That removes people behind the glass wall.
   - `n[t]` is the number of people per frame.
3. **Qwen3-VL-4B grounding** runs on the middle frame (`src/ground.py`) and returns boxes for `safe`, `door` and `desk` in 0–1000 coordinates, then normalized.
   - Desk crop: the grounded desk box widened by 0.05 on each side, extended upward by 1.2 × its height (to include hands and cash), and upscaled 2×.
   - Without a desk box, fall back to the fixed area y 0.30–0.62.
4. **Qwen3-VL-4B desk questions** (`state.ask_frames`, batch 4, `prompts/desk_system_v1.txt` and `prompts/desk_v1.txt`).
   - One image per prompt, answered as JSON 0/1 for seven keys: `id_card, cash, counting, receipt, personal_item, cash_bag, typing`.
   - Batched with left padding.
5. **Qwen3-VL-4B safe check** on each frame's grounded safe crop (`src/safe.py`, constant `Q`).
   - Multiple choice: A = safe open, B = closed, C = not visible. Batch 8, `max_new_tokens=3`.
6. **Gemma-3-4B-it** answers the same seven desk questions on the same crops (batch 6). Gemma's `personal_item` and `cash` / `counting` answers are the ones used.
7. **Pixel features**, no model:
   - **Screen ratio `sr`** = (laptop screen brightness at the end / at the start) ÷ (whole-frame brightness at the end / at the start). Each is averaged over the first and last 3 frames, using the top 60% of the laptop box. `sr < 0.75` means the screen was locked or turned off.
   - **`on`** = mean screen brightness over the first 6 frames. `on ≥ 120` means the screen was on.
   - **Door residual `rs`**:
     - Take grey, blurred frames at 232 px wide, mean-normalized to 128.
     - Average the first 2 frames and the last 2 frames, and diff the two averages.
     - Mask out the person and laptop boxes of those frames.
     - `rs` is the fraction of pixels with |diff| > 25. A value in 0.015 < rs < 0.5 means the door was left open or the scene changed.
8. **Policy** (`src/events.py::events`, thresholds `T = dict(T_client=0.2, T_pers=0.35, T_door=0.015, T_lock=0.75, bag=3)`). A video is a violation if any check fires:

| Check | Condition | Rule | Timestamp |
|---|---|---|---|
| Client session | fraction of frames with n ≥ 2 is ≥ 0.20 | 231.19.1 (no ID shown before cash) or 231.20.1 (a second client visit while cash is still out); otherwise a generic 231.19.1 | first Qwen id_card/cash/counting frame + 3 s |
| Personal items | longest run of Gemma `personal_item` ≥ 0.35 × number of frames | 231.12.1 | time the cashier leaves, otherwise the start of the run |
| Door | 0.015 < rs < 0.5 | 14.08.1 | first frame with a person + 3 frames |
| Laptop left unlocked | the room is empty at the end, on ≥ 120, and not sr < 0.75 | 53.26.1 (source still to be confirmed) | time the cashier leaves |
| Bag left behind | the room is empty at the end, and Qwen `cash_bag` is true in ≥ 3 frames | 231.20.1 | time the cashier leaves |
| Safe left open | ≥ 2 of the last 3 safe answers are "A" | 231.20.1 | start of the final open run |

   - "Time the cashier leaves" is the midpoint between the last seated frame (box centre within 0.06 of its start position) and the first of two consecutive empty frames.
9. **Output.** For each event, take the frame at time t with `frames.frame_at` and encode it as base64 JPEG. The rule title and source come from `data/rules.json`. Verdict: `violation` if there is at least one event.

### Time and VRAM
- Total 40–90 s per video, maximum 99 s, measured on 40 originals and 40 augmented videos.
- The breakdown is returned in `debug.timing_sec` (yolo / qwen / gemma / total).
- The models load one at a time: YOLO on the CPU, then Qwen 4B in bf16, then Gemma 4B in bf16. Peak VRAM is below 12 GB.
- Use `torch.nn.attention.sdpa_kernel` with EFFICIENT + CUDNN backends, because the math backend runs out of memory on Qwen. Gemma also needs MATH added (`vlm.backends`).

### How to rebuild
- Environment: Python 3.12 venv, torch 2.11 + cu128 (needed for sm_120 / RTX 50xx), transformers 5.20, ultralytics 8.4, fastapi, uvicorn, opencv, scikit-learn.
- Weights: `models/Qwen3-VL-4B-Instruct` and `models/gemma-3-4b-it` (Gemma is gated, so accept the licence on Hugging Face). `yolo11m.pt` downloads automatically.
- Run one video: `python src/detect.py video.mov --out res.json`.
- Run the service: `API_KEY=... python -m uvicorn app:app --app-dir src --port 8000`. The endpoints are POST /api/v1/analyze and GET /api/v1/jobs/{id} with the `X-API-Key` header; the UI is at `/`.
- Evaluate: `python src/run_detect.py` (originals) and `python src/run_detect.py --aug` (after `python src/augment.py`).
- Tune thresholds on precomputed features (fast, CPU only): `python src/policy_v5.py gemma4bg v7` runs a leave-one-out grid search. `python src/events.py` scores the timestamps.

### Known weak spots
- **False positives:**
  - Four clean client sessions (Х-3, Х-5, Х-16, Х-19). Clearing them would lose Х-1, Х-4 and Х-18.
  - Х-12, plus one borderline lock case (Х-29) or Х-31.
- **Х-44 (231.41.1):** the cash was checked on only one of the two machines. No detector exists for this; the label was corrected from "mixed denominations".
- **Timestamps:**
  - 16 of the 24 violation videos have a prediction within ±2 s.
  - 19 of the 33 labelled violations match on time (±5 s) and rule.
  - The time offsets were tuned on these same labels.

---

## H2: the same policy end to end, checked under a camera-angle change
- This is H1 run on raw video with no cache.
- To estimate how robust it is to the jury's different angle, `src/augment.py` writes `data/aug/aug_i.mov`: horizontal flip, 4° rotation, then a crop of 5–90% height and 10–95% width (mp4v codec).
- Results: recall 23/24, precision 0.79, F2 0.920. The only miss is Х-44, which the originals caught only by luck.
- What makes it robust: the safe, door and desk regions are grounded by Qwen on each video rather than hard-coded, people are filtered by box position, and the pixel features are ratios normalized for brightness.

---

## H3: classic ML on the features from the models
- `src/ml.py`:
  - `video_features` builds video-level features: the fraction, longest run, first and last values of each per-frame key, people counts and pixel features.
  - `matrix(models)` selects feature columns with a regex.
- Learners:
  - `Cover`, a greedy OR of single-feature thresholds that maximizes F2.
  - LogisticRegression, RandomForest and ExtraTrees.
- All are scored by leave-one-out (`oof`, `report`). `src/groups.py` gives the AUC for each feature group.
- Result: worse than the hand-written OR-policy. There are 24 violation videos across 12 rule types, so a learner overfits.
- Useful as an analysis tool to find which features separate the classes; the H1 thresholds came from this search.
- Time: under 1 s to train once the features are cached.

---

## What failed (skip these)
- **One-prompt few-shot VLM over the whole video:** hallucinated, saying "door left open" almost everywhere, and missed context.
- **Rolling narration (describe frame by frame plus the previous text) with a text judge:** the narration repeated the prompt template. The Qwen3-8B judge (`src/judge.py`) was never fully run.
- **One VLM call per rule (cascade):** too slow.
- **Gemma multiple choice on the laptop screen:** answers "A" everywhere.
- **Gemma on the safe, dark-pixel safe check, standing aspect ratio:** too noisy.
- **Gemma 12B (bnb-4bit mirror) and Qwen 8B:** worse than the 4B models.
- **Mixed-denominations probe (`src/denom.py`):** blind at both 464 px and 1280 px.

## Next steps
1. Run a text-only judge (Qwen3-8B) only on flagged videos to remove clean client sessions without losing recall.
2. Choose the violation type from the event timeline to raise time + rule accuracy.
3. Use a local LLM to extract rules from the regulatory documents (not started). Also confirm the source of rule 53.26.1.
