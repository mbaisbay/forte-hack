"""Run labelled dev clips through the running API and print AGGREGATE metrics only.

Per-clip details (which may describe competition footage) go to a local, git-ignored file
for a teammate to inspect; they are never printed, so a cloud coding agent only sees numbers.

Usage:
  python scripts/eval_dev.py --videos open/videos --labels open/open_labels.xlsx [--subset dev|holdout|all]
"""

import argparse
import hashlib
import json
import re
import statistics
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import openpyxl

# Clips that look like the same recording (identical durations) share a split group.
SAME_GROUP = {"X-47": "X-4", "X-48": "X-5"}
HOLDOUT_SHARE = 0.25


def norm_id(name: str) -> str:
    """'Х-17.mov' (Cyrillic) and 'X-17' (Latin) -> 'X-17'."""
    stem = unicodedata.normalize("NFC", Path(str(name)).stem).strip()
    return stem.replace("Х", "X").replace("х", "X").replace(" ", "")


def parse_time(text) -> float | None:
    if text is None:
        return None
    s = str(text).lower()
    mins = re.search(r"(\d+)\s*мин", s)
    secs = re.search(r"(\d+(?:[.,]\d+)?)\s*сек", s)
    if not mins and not secs:
        raise ValueError(f"cannot parse timestamp label: {text!r}")
    return (int(mins.group(1)) * 60 if mins else 0) + (float(secs.group(1).replace(",", ".")) if secs else 0)


def load_labels(path: Path) -> dict[str, dict]:
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))[1:]
    labels = {}
    for name, ts, verdict in rows:
        if not name:
            continue
        is_violation = str(verdict).strip().lower().startswith("нарушение")
        labels[norm_id(name)] = {"violation": is_violation, "time": parse_time(ts) if is_violation else None}
    return labels


def holdout_ids(labels: dict[str, dict]) -> set[str]:
    """Deterministic split, stratified by label; clips of the same recording stay together."""
    group = lambda i: SAME_GROUP.get(i, i)
    rank = lambda g: hashlib.sha256(g.encode()).hexdigest()
    out: set[str] = set()
    for cls in (True, False):
        groups = sorted({group(i) for i, v in labels.items() if v["violation"] == cls}, key=rank)
        picked = set(groups[: round(HOLDOUT_SHARE * len(groups))])
        out |= {i for i, v in labels.items() if v["violation"] == cls and group(i) in picked}
    return out


def run_clip(base: str, key: str, path: Path, timeout: float) -> dict:
    t0 = time.monotonic()
    with httpx.Client(timeout=120, trust_env=False) as c:
        with path.open("rb") as f:
            r = c.post(f"{base}/api/v1/analyze", headers={"X-API-Key": key}, files={"file": (path.name, f, "video/quicktime")})
        if r.status_code != 200:
            return {"status": "rejected", "http": r.status_code, "sec": time.monotonic() - t0}
        job_id = r.json()["job_id"]
        while time.monotonic() - t0 < timeout:
            body = c.get(f"{base}/api/v1/jobs/{job_id}", headers={"X-API-Key": key}).json()
            if body["status"] != "processing":
                body["sec"] = time.monotonic() - t0
                return body
            time.sleep(1)
    return {"status": "timeout", "sec": time.monotonic() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path("open/videos"))
    ap.add_argument("--labels", type=Path, default=Path("open/open_labels.xlsx"))
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default=None, help="defaults to API_KEY from .env")
    ap.add_argument("--subset", choices=["dev", "holdout", "all"], default="dev")
    ap.add_argument("--parallel", type=int, default=1, help="concurrent submissions (simulate the jury)")
    ap.add_argument("--details", type=Path, default=Path("eval_details.json"))
    ap.add_argument("--results-dir", type=Path, default=Path("data/results"),
                    help="where the worker writes <job>.meta.json (counts and timings only)")
    args = ap.parse_args()

    key = args.key
    if key is None:
        env = dict(l.split("=", 1) for l in Path(".env").read_text().splitlines() if "=" in l and not l.startswith("#"))
        key = env["API_KEY"]

    labels = load_labels(args.labels)
    clips = {norm_id(p.name): p for p in sorted(args.videos.glob("*.mov"))}
    missing = sorted(set(labels) - set(clips))
    holdout = holdout_ids(labels)
    in_subset = lambda i: args.subset == "all" or (i in holdout) == (args.subset == "holdout")
    ids = sorted(i for i in labels if i in clips and in_subset(i))

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        outs = dict(zip(ids, pool.map(lambda i: run_clip(args.base, key, clips[i], 330), ids)))

    tp = fp = tn = fn = failed = 0
    errs, secs, n_events = [], [], []
    details = {}
    diag: dict[str, list[dict]] = {}
    for i in ids:
        lab, out = labels[i], outs[i]
        secs.append(out["sec"])
        if out.get("status") != "done":
            failed += 1
            details[i] = {"label": lab, "status": out.get("status"), "error": out.get("error")}
            # An unprocessed clip counts as wrong either way.
            fn += lab["violation"]; fp += not lab["violation"]
            continue
        pred = out["verdict"] == "violation"
        outcome = ("TP" if lab["violation"] else "FP") if pred else ("FN" if lab["violation"] else "TN")
        meta_path = args.results_dir / f"{out['job_id']}.meta.json"
        if meta_path.exists():
            diag.setdefault(outcome, []).append(json.loads(meta_path.read_text()))
        times = [v["timecode_sec"] for v in out["violations"]]
        n_events.append(len(times))
        if pred and lab["violation"]:
            tp += 1
            errs.append(min(abs(t - lab["time"]) for t in times))
        elif pred:
            fp += 1
        elif lab["violation"]:
            fn += 1
        else:
            tn += 1
        details[i] = {"label": lab, "verdict": out["verdict"], "sec": round(out["sec"], 1),
                      "events": [{"type": v["type"], "t": v["timecode_sec"], "description": v["description"]} for v in out["violations"]]}

    args.details.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(ids)
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "n/a"
    print(f"subset={args.subset} clips={n} parallel={args.parallel} (labels without video: {len(missing)})")
    print(f"video-level: TP={tp} FP={fp} TN={tn} FN={fn} failed/timeout={failed}")
    print(f"accuracy={pct(tp + tn, n)} recall={pct(tp, tp + fn)} precision={pct(tp, tp + fp)}")
    if errs:
        print(f"timecode |error| on TP: median={statistics.median(errs):.1f}s max={max(errs):.1f}s "
              f"<=1s:{pct(sum(e <= 1 for e in errs), len(errs))} <=2s:{pct(sum(e <= 2 for e in errs), len(errs))} "
              f"<=3s:{pct(sum(e <= 3 for e in errs), len(errs))}")
    if n_events:
        print(f"events per clip: mean={statistics.mean(n_events):.2f} max={max(n_events)}")
    if secs:
        print(f"end-to-end sec: median={statistics.median(secs):.0f} max={max(secs):.0f} over_300={sum(s > 300 for s in secs)}")
    for outcome in ("TP", "FN", "FP", "TN"):
        metas = diag.get(outcome, [])
        if not metas:
            continue
        status_tot: dict[str, int] = {}
        for m in metas:
            for k, v in m.get("final_status_counts", {}).items():
                status_tot[k] = status_tot.get(k, 0) + v
        no_cand = sum(m.get("raw_candidates", 0) == 0 for m in metas)
        print(f"diag {outcome} n={len(metas)}: clips with no pass-1 candidate={no_cand} "
              f"mean raw cands={statistics.mean(m.get('raw_candidates', 0) for m in metas):.1f} "
              f"mean max pass-1 conf={statistics.mean(m.get('max_pass1_conf', 0) for m in metas):.2f} "
              f"final statuses={status_tot}")
    print(f"per-clip details written locally to {args.details} (do not paste into cloud tools)")


if __name__ == "__main__":
    main()
