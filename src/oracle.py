"""Oracle test: can the VLM see a violation when given the exact labeled window + the exact rule?

For every GT event: clip [t-W, t+W] + that rule's question -> yes/no (should be yes).
Negative control: same rule question on a window of a clean video at the same time -> should be no.
Also asks a free description of each positive clip for manual reading.

  python src/oracle.py --model models/Qwen3-VL-4B-Instruct --fps 2 --width 448 --win 4
"""
import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import frames, vlm
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
SYS = ("You review overhead CCTV of a small bank cashier room (desk with laptop and counting machine, grey safe, glass door). "
       "The cashier works behind the desk on the laptop/safe side; clients/couriers come through the door. "
       "Frames are in time order, each preceded by its timestamp.")


def clip(v, t, a):
    fr = frames.sample(video_path(v), fps=a.fps, max_frames=10_000, width=a.width, overlay=True)
    return [x for x in fr if t - a.win <= x[0] <= t + a.win]


def content(fr):
    c = []
    for t, im in fr:
        c += [f"t={t:.1f}s", im]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--win", type=float, default=4)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--out", default="runs/oracle.json")
    a = ap.parse_args()
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    rules = {r["id"]: r for r in json.loads((ROOT / "data/rules.json").read_text(encoding="utf-8"))}
    clean = [v for v, g in gt.items() if not g]
    random.seed(0)
    rows = []
    for v, evs in gt.items():
        for e in evs:
            rid = next((r for r in e["rules"] if r in rules), None)
            if not rid:
                continue
            q = (f"Rule [{rid}] {rules[rid]['title']}: {rules[rid]['text']}\nQuestion: {rules[rid]['check']}\n"
                 'Answer JSON: {"answer": "yes"|"no", "reason": "<one sentence>"}')
            neg = random.choice(clean)
            for kind, vid, t in [("pos", v, e["t"]), ("neg", neg, e["t"])]:
                fr = clip(vid, t, a)
                if not fr:
                    continue
                t0 = time.time()
                out, ntok = vlm.generate(a.model, a.quant, SYS, content(fr) + [q], 80)
                row = dict(kind=kind, video=vid, t=t, rule=rid, frames=len(fr), tokens=ntok,
                           sec=round(time.time() - t0, 1), yes="yes" in out.lower()[:40], out=out.strip())
                if a.describe and kind == "pos":
                    d, _ = vlm.generate(a.model, a.quant, SYS, content(fr) + [
                        "Describe step by step what each person does with cash, documents, the safe, the laptop and the door."], 200)
                    row["describe"] = d.strip()
                rows.append(row)
                print(f"{kind} {vid:6s} t={t:<4} {rid:9s} yes={row['yes']!s:5} {row['sec']}s {ntok}tok | {out.strip()[:110]}", flush=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    pos = [r for r in rows if r["kind"] == "pos"]; neg = [r for r in rows if r["kind"] == "neg"]
    tpr = sum(r["yes"] for r in pos) / len(pos); fpr = sum(r["yes"] for r in neg) / len(neg)
    print(f"\nORACLE  recall(yes on true window)={tpr:.2f}  false-yes on clean={fpr:.2f}  "
          f"sec/call={sum(r['sec'] for r in rows)/len(rows):.1f}")
    by = {}
    for r in rows:
        by.setdefault(r["rule"], [0, 0, 0, 0])
        i = 0 if r["kind"] == "pos" else 2
        by[r["rule"]][i] += r["yes"]; by[r["rule"]][i + 1] += 1
    for k, (py, pn, ny, nn) in sorted(by.items()):
        print(f"  {k:9s} pos {py}/{pn}  neg {ny}/{nn}")


if __name__ == "__main__":
    main()
