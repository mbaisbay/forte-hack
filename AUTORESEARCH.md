# Autoresearch: cash-desk violation detector

You are an autonomous research agent. Goal: maximize `score` on the **train** split of the
violation detector, without overfitting, by iterating on prompts, rule descriptions and config.

## Setup (read once)
- `src/pipeline.py` - video -> frames -> local Qwen-VL -> violations. Modes: `all` (one prompt, all rules) and `per_rule` (one yes/no prompt per rule).
- `src/evaluate.py` - runs a config on a split, prints per-video gt vs pred, appends a row to `runs/results.tsv`.
- `data/rules.json` - rule catalogue: id, source, title, text, `check` (visual question), `fewshot` (text examples from train videos).
- `prompts/*.txt` - system prompt and prompt templates (`{RULES}`, `{RULE}`, `{DURATION}` placeholders).
- `configs/*.json` - experiment knobs: model, quant, mode, fewshot, fps, max_frames, width, overlay, text_timestamps, max_new_tokens, min_conf, dedup_sec, disabled_rules.
- `data/gt.json` - ground truth (do NOT edit). Labels are partly noisy; timestamps are approximate.

Run: `PYTHONUTF8=1 .venv/Scripts/python src/evaluate.py configs/<cfg>.json --split train --name <exp> --note "<what changed>"`
Quick check: add `--limit 6`. Runs resume if interrupted (same `--name`).

## Metric
`score = 0.4*verdict_acc + 0.3*event_f1 + 0.3*typed_f1` (event = predicted timecode within 5s of a GT event; typed = also same rule paragraph, e.g. 231.20).
Constraint: `sec_per_video` must stay < 120 (jury timeout is 300s per video on a possibly slower server).

## Loop
1. Read `runs/results.tsv` and the latest run's per-video output (`runs/<exp>/preds.json`, incl. `raw` model text). Find the dominant error type: false positives on clean videos? missed rules? wrong timecodes? bad JSON?
2. Form ONE hypothesis and make ONE focused change (new prompt file `prompts/<name>_vN.txt` + new config `configs/<exp>.json`; never overwrite a file used by an earlier run, so results stay reproducible).
3. Run on train. Compare to the best run so far.
4. Keep the change if score improves (>= +0.01) or equal score with lower sec_per_video; otherwise revert to the best config.
5. Log one line to `runs/journal.md`: exp name, hypothesis, result, keep/discard, next idea.
6. Every ~5 kept improvements, run the best config on `--split dev` (held-out, 14 videos) and log it. Never tune on dev. If dev drops while train rises, you are overfitting - back off.
7. Repeat. Do not stop to ask the user; stop after 25 experiments or when 6 in a row fail to improve, then write a summary at the end of `runs/journal.md`.

## Idea backlog (try roughly in this order)
- `all` vs `per_rule` mode (baseline both first).
- few-shot on/off; rewrite `check` questions to describe concrete visual cues (what objects, where on screen).
- Require a short "observation log" before the JSON (chain-of-thought) vs JSON only.
- Ask for confidence and tune `min_conf` to cut false positives on clean videos.
- Two-stage: first describe the video timeline (who enters, what is on the desk, door state, laptop state) as text, then judge the rules on that text.
- fps 1 vs 2, width 320 vs 448, max_frames 32/48/64.
- Model: Qwen3-VL-4B bf16 vs Qwen3-VL-8B 4bit (`"quant": "4bit"`).
- Merge/disable rules that are never hit, or that cause most false positives.

## Rules
- Do not edit `data/gt.json` or `src/evaluate.py` scoring.
- Few-shot texts may only describe **train** videos (dev videos are listed in `evaluate.py: DEV`).
- Keep everything local; no cloud APIs.
