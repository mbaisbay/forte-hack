# ForteBank Datathon 26: cashier video violation detector

A local AI service that analyses cashier CCTV videos. It reports possible cash-handling violations with a timecode, evidence frames and a reference to the regulation (Kazakhstan NBRK rules). It runs on one 12 GB GPU with pretrained open models only (YOLO11, Qwen3-VL-4B, Gemma-3-4B, Qwen3-8B), and no cloud AI.

Branch `nursultan-chokushev`: this approach is independent of `main`.

| Path | What it is |
|---|---|
| [`service/`](service/) | **The deliverable.** Web UI (videos + labeling, rules catalogue + LLM rule extraction, editable decision tree) and the jury API. Start with [`service/README.md`](service/README.md). Agent and developer guide: [`service/AGENTS.md`](service/AGENTS.md). |
| [`final_results.md`](final_results.md) | Best approaches, measured quality, timings, and how to rebuild them. |
| [`runs/journal.md`](runs/journal.md) | Full experiment log: what was tried, what worked, what failed. |
| `src/`, `prompts/`, `configs/` | Research code: feature extraction, policy search, evaluation, augmentation. |

Not in the repository (hackathon data rules and size): videos, labels, organiser documents, model weights and runtime storage.

## Results (40 training videos)

**Verdict**

| Run | Recall | Precision | Accuracy | F2 |
|---|---|---|---|---|
| Decision tree on precomputed features | 24/24 | 0.83 | 87.5% | 0.960 |
| Full service, original videos | 24/24 | 0.80 | 85.0% | 0.952 |
| Full service, simulated new camera angle | 23/24 | 0.79 | 82.5% | 0.920 |

A cashier working alone is not flagged by itself. The checks look at what she leaves behind (cash, bag, open safe, unlocked laptop).

**Speed:** about 53 s per video on average, at most about 100 s; the jury limit is 5 minutes.

**Rule extraction from the catalogue** (Qwen3-8B, local):
- 24/24 paragraphs extracted
- 10/12 of our rules recognised as visible on camera
- 90 s per document
- references built by code, so none are invented

## Quick start
```
cd service
pip install -r requirements.txt && python download_models.py
API_KEY=<secret> python -m uvicorn app:app --app-dir src --port 8000      # open http://localhost:8000
```
