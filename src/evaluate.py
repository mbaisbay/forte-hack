"""Run a config on a split and score it.

  python src/evaluate.py configs/base_all.json --split train --name exp01

Few-shot examples of the evaluated video are always hidden (leave-one-out).
Appends one row to runs/results.tsv; per-video predictions in runs/<name>/preds.json.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pipeline

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "open/open/videos"
DEV = ["Х-17", "Х-37", "Х-20", "Х-26", "Х-28", "Х-40", "Х-32", "Х-44",
       "Х-3", "Х-10", "Х-14", "Х-21", "Х-27", "Х-43"]
TOL = 5.0


def video_path(v):
    p = VIDEOS / f"{v}.mov"
    return p if p.exists() else VIDEOS / f"{v.replace('Х', 'X')}.mov"


def para(rule):  # "231.20.1" -> "231.20"
    return ".".join(rule.split(".")[:2])


def match(preds, gts, typed):
    used, tp = set(), 0
    for p in sorted(preds, key=lambda p: p["timecode_sec"]):
        for i, g in enumerate(gts):
            if i in used or abs(p["timecode_sec"] - g["t"]) > TOL:
                continue
            if typed and para(p["rule_id"]) not in {para(r) for r in g["rules"]}:
                continue
            used.add(i); tp += 1
            break
    return tp


def f1(tp, np_, ng):
    p = tp / np_ if np_ else (1.0 if ng == 0 else 0.0)
    r = tp / ng if ng else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(preds, gt):
    c = dict(tp=0, fp=0, fn=0, tn=0)
    ev = dict(np=0, ng=0, tp=0, tpt=0)
    for v, g in gt.items():
        if v not in preds:
            continue
        pv = preds[v]["violations"]
        key = ("t" if g else "f") + ("p" if pv else "n")
        c[{"tp": "tp", "tn": "fn", "fp": "fp", "fn": "tn"}[key]] += 1
        ev["np"] += len(pv); ev["ng"] += len(g)
        ev["tp"] += match(pv, g, False); ev["tpt"] += match(pv, g, True)
    n = sum(c.values())
    acc = (c["tp"] + c["tn"]) / n
    _, _, vf1 = f1(c["tp"], c["tp"] + c["fp"], c["tp"] + c["fn"])
    ep, er, ef1 = f1(ev["tp"], ev["np"], ev["ng"])
    tp_, tr_, tf1 = f1(ev["tpt"], ev["np"], ev["ng"])
    return dict(n=n, verdict_acc=round(acc, 3), verdict_f1=round(vf1, 3), **c,
                event_p=round(ep, 3), event_r=round(er, 3), event_f1=round(ef1, 3),
                typed_p=round(tp_, 3), typed_r=round(tr_, 3), typed_f1=round(tf1, 3),
                score=round(0.4 * acc + 0.3 * ef1 + 0.3 * tf1, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="train", choices=["train", "dev", "all"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    vids = {"dev": [v for v in gt if v in DEV], "train": [v for v in gt if v not in DEV], "all": list(gt)}[a.split]
    if a.limit:
        vids = vids[:a.limit]
    name = a.name or f"{Path(a.config).stem}_{a.split}_{time.strftime('%m%d_%H%M%S')}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    pf = out / "preds.json"
    preds = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
    for v in vids:
        if v in preds:
            continue
        preds[v] = pipeline.analyze(video_path(v), cfg, exclude_fewshot=[v])
        g = gt[v]
        print(f"{v:6s} gt={[(x['t'], x['rules']) for x in g]} pred={[(p['timecode_sec'], p['rule_id']) for p in preds[v]['violations']]} "
              f"{preds[v]['sec']}s", flush=True)
        pf.write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    m = score({v: preds[v] for v in vids}, {v: gt[v] for v in vids})
    m["sec_per_video"] = round(sum(preds[v]["sec"] for v in vids) / len(vids), 1)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(m))
    tsv = ROOT / "runs/results.tsv"
    cols = ["score", "verdict_acc", "verdict_f1", "event_f1", "typed_f1", "event_p", "event_r", "sec_per_video", "n"]
    if not tsv.exists():
        tsv.write_text("name\tsplit\t" + "\t".join(cols) + "\tnote\n", encoding="utf-8")
    with tsv.open("a", encoding="utf-8") as f:
        f.write(f"{name}\t{a.split}\t" + "\t".join(str(m[c]) for c in cols) + f"\t{a.note}\n")


if __name__ == "__main__":
    main()
