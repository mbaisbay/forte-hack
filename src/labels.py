"""Parse labels_fixed.xlsx into data/gt.json: {video: [{"t": sec, "rule": "231.20.1"}...]}"""
import json, re, sys
import pandas as pd

def norm(name):  # Latin X vs Cyrillic Х in filenames
    return str(name).strip().replace("X", "Х")

def parse_t(s):
    m = re.search(r"(\d+)\s*мин", s); mins = int(m.group(1)) if m else 0
    sec = re.search(r"(\d+)\s*сек", s)
    return mins * 60 + (int(sec.group(1)) if sec else 0)

def main(path="labels_fixed.xlsx", out="data/gt.json"):
    df = pd.read_excel(path)
    df.columns = ["video", "t", "label", "rule", "note"]
    gt = {}
    for r in df.itertuples():
        if pd.isna(r.video):
            continue
        v = norm(r.video)
        gt.setdefault(v, [])
        if r.label == "нарушение":
            rules = [x.strip() for x in str(r.rule).split(",")]
            gt[v].append({"t": parse_t(str(r.t)), "rules": rules, "note": str(r.note)})
    # byte-identical duplicate files: Х-47 == Х-4, Х-48 == Х-5 -> merge labels, drop copies
    for dup, orig in {"Х-47": "Х-4", "Х-48": "Х-5"}.items():
        gt[orig] += gt.pop(dup, [])
    for v, evs in gt.items():  # drop identical (t, rules) events
        seen, uniq = set(), []
        for e in evs:
            k = (e["t"], tuple(e["rules"]))
            if k not in seen:
                seen.add(k); uniq.append(e)
        gt[v] = uniq
    json.dump(gt, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(gt), "videos,", sum(1 for v in gt.values() if v), "with violations,",
          sum(len(v) for v in gt.values()), "events")

if __name__ == "__main__":
    main(*sys.argv[1:])
