"""Run detect.analyze on all original videos (--aug: augmented copies) and score verdict + timestamps."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import detect, ml
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
aug = "--aug" in sys.argv
out = ROOT / ("runs/detect_aug.json" if aug else "runs/detect_orig.json")
res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
items = (list(json.loads((ROOT / "data/aug/map.json").read_text(encoding="utf-8")).items()) if aug
         else [(None, v) for v in ml.VIDS])
for fn, v in items:
    if v in res:
        continue
    p = ROOT / "data/aug" / fn if aug else video_path(v)
    r = detect.analyze(p)
    res[v] = {"verdict": r["verdict"], "events": [(x["rule_id"], x["timecode_sec"]) for x in r["violations"]],
              "sec": r["debug"]["timing_sec"]["total"]}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(v, r["verdict"], res[v]["events"], res[v]["sec"], flush=True)
Y = {v: bool(ml.GT[v]) for v in res}
tp = sum(res[v]["verdict"] == "violation" and Y[v] for v in res); fp = sum(res[v]["verdict"] == "violation" and not Y[v] for v in res)
npos = sum(Y.values()); pr = tp / max(1, tp + fp); rc = tp / max(1, npos)
print(f"{'AUG' if aug else 'ORIG'}: recall {tp}/{npos} FP {fp}/{len(res) - npos} F2 {5 * pr * rc / max(1e-9, 4 * pr + rc):.3f} "
      f"missed {[v for v in res if Y[v] and res[v]['verdict'] != 'violation']} "
      f"FP {[v for v in res if not Y[v] and res[v]['verdict'] == 'violation']} max sec {max(r['sec'] for r in res.values())}")
