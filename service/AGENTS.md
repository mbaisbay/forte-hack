# AGENTS.md: guide for coding agents working on this service

A local web service that analyses cashier CCTV videos (ForteBank Datathon 26). It finds possible cash-handling violations and returns the verdict, timecode, 1–3 evidence frames and the rule source. User-facing text is in Russian; code and comments are in English. The case requirements are in `../Кейс_ForteBank.docx`; the API contract is in section 10, so do not change its field names.

## Layout
```
src/app.py          FastAPI: pages + jury API (/api/v1, X-API-Key) + UI API (/api/ui, no key)
src/worker.py       the only GPU consumer: one thread, PriorityQueue (videos=0 before doc paragraphs=1), re-queues unfinished jobs on start
src/db.py           SQLite storage/service.db (videos, docs); JSON helpers for data/*.json (keeps one .bak.json)
src/detect.py       analyze(path) -> {"status","verdict","violations",debug}; extracts all signals, then events.events(F, tree)
src/events.py       decision tree interpreter: signals F + data/tree.json -> [(node_id, t, message, rule_ids)]
src/rules_llm.py    document -> units() (code-built sources) -> extract() with Qwen3-8B 4-bit -> suggestions
src/eval_rules.py   rule-extraction quality on the catalogue PDF -> data/eval_rules_report.json (shown on /rules)
src/frames.py vlm.py state.py ground.py safe.py residual.py    model and pixel helpers
src/static/         videos.html (/), rules.html (/rules), tree.html (/tree), common.css; plain JS, no build step
data/rules.json     rule catalogue: id, source, title, text, check, fewshot, [origin]
data/tree.json      decision tree: params (global) + nodes {id,parent,title,observe,params,enabled,rule_ids,message}
prompts/            desk yes/no prompts (7 keys) for Qwen and Gemma
storage/            runtime data (db, uploaded videos and documents); not source, safe to wipe when the service is stopped
```

## Run and test
```
pip install -r requirements.txt && python download_models.py          # or set MODELS_DIR to existing weights
API_KEY=secret python -m uvicorn app:app --app-dir src --port 8000
python src/detect.py video.mov                                       # one video, no server
python src/eval_rules.py ../Каталог_правил_кассовые_операции_Datathon_26_1.pdf [--cached]
```
- **Smoke test without the GPU:** `fastapi.testclient.TestClient(app.app)` against `/api/ui/*`. Importing `app` starts the worker thread, which is idle until a job arrives.
- **API checks:** POST without a key → 401; unknown job → 404; bad tree or duplicate rule ids → 400.
- **Policy regression** (no GPU, the main repo has cached features in `../data`): run `events.events(F, tree)` on `../src/events.cached(v)` for all 40 videos and compare recall / FP / F2 before and after any tree or `events.py` change. Recall must stay 24/24.
- **Windows:** set `PYTHONUTF8=1`. Video names are Cyrillic; `frames._open` copies them to an ASCII temp path for cv2.

## How a verdict is made (keep this mental model)
1. **Sample frames.** `frames.sample`: about 1 fps over the whole video, at most 60 frames, 464 px wide.
2. **Per-frame signals.**
   - YOLO11m, CPU: people (box bottom y2 > `params.person_min_y` counts as at the desk) and the laptop.
   - Qwen3-VL-4B:
     - grounds safe / door / desk on the middle frame
     - answers 7 desk yes/no keys per frame (`Q`)
     - answers safe open / closed / hidden per frame (`sf`)
   - Gemma-3-4B: the same 7 keys (`D`).
   - Pixels: screen ratio `sr`, screen-on `on`, door residual `rs`.
3. **Decision tree.** `events.py` walks `tree.json`. A child node runs only if its parent is enabled and fired. The video is a violation if any enabled leaf fires. The leaf's first rule id is the violation `type`; all rule ids form `source`.
4. **Output.** `detect.py` adds 3 evidence frames (t, t−2, t+2) and the rule title and source from `rules.json`. Both JSON files are re-read on every video, so UI edits apply without a restart.

## Adding or changing a check
- **Threshold, wording or rule mapping:** edit `data/tree.json` (or use `/tree`). No code change.
- **New check:**
  1. Add the node to `tree.json` (unique id; `params` are the numbers you read).
  2. Add a block in `events.py`: `if on("<id>") and <condition>: add("<id>", t)`.
  3. If it needs a new per-frame signal, add it in `detect.analyze` and put it into `F`.
  4. Re-run the policy regression.
- `rules_llm.leaf_nodes` lists the leaves the LLM may map rules to. `client_fallback` is excluded on purpose.
- **Keep the priority order:** no missed violations (recall) first, then few false alarms, then timecodes, then type. The case FAQ says a miss is unacceptable. `client_fallback` and `solo_cash` are recall safety nets; don't remove them without measuring.

## GPU and memory rules (12 GB)
- Only `worker.py` touches models. Do not call `detect` or `rules_llm` from request handlers.
- `vlm.load` keeps one model and swaps on demand. `rules_llm.load()` calls `vlm.free()`, and video jobs call `rules_llm.free()` first. Keep this pairing when adding models.
- **Attention kernels:** use `sdpa_kernel(vlm.backends(model))`. The math kernel runs out of memory on Qwen; Gemma needs it added.
- **Time limit:** a jury video must finish in under 5 minutes; it is about 55 s on average now. Document paragraphs are queued one by one so they never block a video for long.

## Rule references
- The case scores "no invented references". Sources from documents are built by `rules_llm.units()` from the document structure, never by the LLM.
- Rule 53.26.1 has an unconfirmed source. Don't add more like it.
- `/api/ui/docs` marks suggestions whose (document number, paragraph) already exists in `rules.json` as `exists`, and the UI hides them.

## Measured baselines (40 training videos; update when you change things)
- **Verdict:** recall 24/24. On precomputed features: precision 0.77, F2 0.945 with `solo_cash` on (0.83 / 0.960 with it off).
- **Full service:** F2 0.952 on the originals (measured before `solo_cash` was added). On a simulated new angle, recall was 23/24 before `solo_cash`; it is expected to fix that miss (Х-44), see `../runs/solo_check.json`.
- **Rule extraction:** 24/24 paragraphs, 10/12 visible rules recognised, 4/5 check mappings, 90 s per document.
- **Experiment history:** `../runs/journal.md` and `../final_results.md`.
