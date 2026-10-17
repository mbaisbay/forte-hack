"""Run labelled dev clips through the running API and print AGGREGATE metrics only.

Per-clip details (which may describe competition footage) go to a local, git-ignored file
for a teammate to inspect; they are never printed, so a cloud coding agent only sees numbers.

Usage:
  python scripts/eval_dev.py --videos open/videos --labels open/labels_fixed.xlsx [--subset dev|holdout|all]
"""

import argparse
import json
import re
import statistics
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import openpyxl

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
    """One clip may have several rows (several violations); optional 4th column holds rule codes like '231.20.1, 231.12.1'."""
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    labels: dict[str, dict] = {}
    for row in list(ws.iter_rows(values_only=True))[1:]:
        name, ts, verdict = row[:3]
        if not name:
            continue
        lab = labels.setdefault(norm_id(name), {"violation": False, "times": [], "codes": []})
        if str(verdict).strip().lower().startswith("нарушение"):
            lab["violation"] = True
            lab["times"].append(parse_time(ts))
            codes = row[3] if len(row) > 3 else None
            lab["codes"] += [c.strip() for c in str(codes or "").split(",") if c.strip() and c.strip() not in lab["codes"]]
    for lab in labels.values():
        lab["times"] = sorted(t for t in lab["times"] if t is not None)
        lab["time"] = lab["times"][0] if lab["times"] else None
    return labels


# Frozen when the first labels arrived (stratified, same-recording clips kept together), so later
# label fixes cannot move tuned-on clips into the holdout.
HOLDOUT = {"X-3", "X-11", "X-14", "X-19", "X-24", "X-26", "X-33", "X-34", "X-35", "X-39"}


def holdout_ids(labels: dict[str, dict]) -> set[str]:
    return HOLDOUT & set(labels)


def rule_codes(rules_path: Path) -> dict[str, set[str]]:
    """Russian type name (as the API returns it) -> label codes it covers."""
    rules = json.loads(rules_path.read_text(encoding="utf-8"))["rules"]
    return {r["type_ru"]: set(r.get("codes", [])) for r in rules}


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
    ap.add_argument("--labels", type=Path, default=Path("open/labels_fixed.xlsx"))
    ap.add_argument("--rules", type=Path, default=Path("config/rules.json"))
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
    codes_of = rule_codes(args.rules)
    clips = {norm_id(p.name): p for p in sorted(args.videos.glob("*.mov"))}
    missing = sorted(set(labels) - set(clips))
    holdout = holdout_ids(labels)
    in_subset = lambda i: args.subset == "all" or (i in holdout) == (args.subset == "holdout")
    ids = sorted(i for i in labels if i in clips and in_subset(i))

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        outs = dict(zip(ids, pool.map(lambda i: run_clip(args.base, key, clips[i], 330), ids)))

    tp = fp = tn = fn = failed = type_hit = 0
    per_code: dict[str, list[int]] = {}  # code -> [clips, detected, detected with matching type]
    errs, secs, n_events = [], [], []
    details = {}
    diag: dict[str, list[dict]] = {}
    meta_by_clip: dict[str, dict] = {}
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
            meta_by_clip[i] = json.loads(meta_path.read_text())
            diag.setdefault(outcome, []).append(meta_by_clip[i])
        times = [v["timecode_sec"] for v in out["violations"]]
        n_events.append(len(times))
        if lab["violation"]:
            pred_codes = set().union(*(codes_of.get(v["type"], set()) for v in out["violations"]))
            for code in lab["codes"]:
                row = per_code.setdefault(code, [0, 0, 0])
                row[0] += 1; row[1] += pred; row[2] += pred and code in pred_codes
        if pred and lab["violation"]:
            tp += 1
            type_hit += bool(pred_codes & set(lab["codes"]))
            if lab["times"]:
                errs.append(min(abs(t - lt) for t in times for lt in lab["times"]))
        elif pred:
            fp += 1
        elif lab["violation"]:
            fn += 1
        else:
            tn += 1
        details[i] = {"label": lab, "job_id": out["job_id"], "verdict": out["verdict"], "sec": round(out["sec"], 1),
                      "events": [{"type": v["type"], "t": v["timecode_sec"], "description": v["description"]} for v in out["violations"]]}

    args.details.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(ids)
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "n/a"
    print(f"subset={args.subset} clips={n} parallel={args.parallel} (labels without video: {len(missing)})")
    print(f"video-level: TP={tp} FP={fp} TN={tn} FN={fn} failed/timeout={failed}")
    print(f"accuracy={pct(tp + tn, n)} recall={pct(tp, tp + fn)} precision={pct(tp, tp + fp)}")
    if tp:
        print(f"type match on TP: {type_hit}/{tp} clips have a predicted type whose code is in the label")
    for code, (n_c, det, typ) in sorted(per_code.items()):
        print(f"  code {code}: clips={n_c} detected={det} right_type={typ}")
    by_type: dict[str, list[int]] = {}  # type -> [clean clips flagged, violation clips right code, violation clips other code]
    for i, det in details.items():
        for typ in {e["type"] for e in det.get("events", [])}:
            row = by_type.setdefault(typ, [0, 0, 0])
            lab = labels[i]
            if not lab["violation"]:
                row[0] += 1
            else:
                row[1 if codes_of.get(typ, set()) & set(lab["codes"]) else 2] += 1
    for typ, (fl, ok, other) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        print(f"  type '{typ}': clean_flagged={fl} right={ok} other_violation={other}")
    if errs:
        print(f"timecode |error| on TP: median={statistics.median(errs):.1f}s max={max(errs):.1f}s "
              f"<=1s:{pct(sum(e <= 1 for e in errs), len(errs))} <=2s:{pct(sum(e <= 2 for e in errs), len(errs))} "
              f"<=3s:{pct(sum(e <= 3 for e in errs), len(errs))}")
    if n_events:
        print(f"events per clip: mean={statistics.mean(n_events):.2f} max={max(n_events)}")
    if secs:
        print(f"end-to-end sec: median={statistics.median(secs):.0f} max={max(secs):.0f} over_300={sum(s > 300 for s in secs)}")
    near = lambda iv, t, pad=2.0: iv[0] - pad <= t <= iv[1] + pad
    viol = [(m, labels[i]["times"]) for i, m in meta_by_clip.items() if labels[i]["violation"] and labels[i]["times"]]
    if viol:
        seen1 = sum(any(near(iv, t) for iv in m.get("pass1_intervals", []) for t in ts) for m, ts in viol)
        kept = sum(any(near(iv, t) and iv[2] == "supported" for iv in m.get("final_intervals", []) for t in ts) for m, ts in viol)
        print(f"label-time hit (±2s): pass-1 flagged near the label in {seen1}/{len(viol)} violation clips; "
              f"re-check confirmed near the label in {kept}/{len(viol)}")
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
