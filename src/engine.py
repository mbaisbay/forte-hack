"""Simple yes/no violation engine over per-frame features.

features per frame: VLM answers (raw from state mode) + n = YOLO people inside room.
  python src/engine.py runs/feat_v1 [--split train]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

TH = dict(smooth=1, away=2, expo=2, door=6, personal=3, outsider=2)


def smooth(x, k):
    return [sum(x[max(0, i - k):i + k + 1]) * 2 > len(x[max(0, i - k):i + k + 1]) for i in range(len(x))]


def longest(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def load(run, v):
    raw = json.loads((ROOT / run / "preds.json").read_text(encoding="utf-8"))[v]["raw"]
    ppl = json.loads((ROOT / "data/people" / f"{v}.json").read_text(encoding="utf-8"))
    n = [sum(1 for b in bs if b[3] > 0.40) for bs in ppl.values()]
    n = (n + [n[-1]] * len(raw))[:len(raw)]  # align lengths (same sampler, should match)
    for f, k in zip(raw, n):
        f["n"] = k
    return raw


def verdict(F, th=TH):
    """Return dict rule_name -> bool (fired)."""
    k = th["smooth"]
    c = lambda key: smooth([bool(f.get(key, 0)) for f in F], k)
    n = [f["n"] for f in F]
    empty = smooth([x == 0 for x in n], k)                 # nobody in the room (YOLO)
    away = [e or not a for e, a in zip(empty, c("cashier_in_room"))]
    laptop, cash, safe, bag = c("laptop_screen_on"), c("cash_on_desk"), c("safe_open"), c("cash_bag")
    client = smooth([x >= 2 for x in n], k)
    fired = {
        # A: valuables exposed while cashier away
        "A_laptop_unattended": longest([a and l for a, l in zip(away, laptop)]) >= th["away"],
        "A_cash_unattended": longest([a and (x or b) for a, x, b in zip(away, cash, bag)]) >= th["expo"],
        "A_safe_unattended": longest([a and s for a, s in zip(away, safe)]) >= th["expo"],
        # C: client session without ID seen
        "C_no_id": longest(client) >= 4 and not any(c("id_document")),
        # D: perimeter
        "D_door_open": longest(c("door_open")) >= th["door"],
        "D_outsider": longest(c("outsider_cashier_side")) >= th["outsider"],
        # E: personal items
        "E_personal": longest(c("personal_item")) >= th["personal"],
    }
    return fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rules", default="", help="comma list of rule names to use (default all)")
    a = ap.parse_args()
    from evaluate import DEV
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    have = json.loads((ROOT / a.run / "preds.json").read_text(encoding="utf-8"))
    vids = [v for v in gt if v in have and (a.split == "all" or (v in DEV) == (a.split == "dev"))]
    use = set(a.rules.split(",")) if a.rules else None
    tp = fp = fn = tn = 0
    per_rule = {}
    for v in vids:
        fired = verdict(load(a.run, v))
        on = [r for r, x in fired.items() if x and (use is None or r in use)]
        y = bool(gt[v])
        tp += y and bool(on); fn += y and not on; fp += (not y) and bool(on); tn += (not y) and not on
        for r in on:
            per_rule.setdefault(r, [0, 0]); per_rule[r][0 if y else 1] += 1
        print(f"{v:6s} {'VIOL ' if y else 'clean'} {'OK ' if y == bool(on) else 'ERR'} {[x['rules'][0] for x in gt[v]]} -> {on}")
    n = len(vids)
    print(f"\nverdict acc {(tp + tn) / n:.2f}  (tp {tp} fn {fn} fp {fp} tn {tn}, n={n})")
    print("rule fired on: [violation videos, clean videos]")
    for r, (a_, b_) in sorted(per_rule.items()):
        print(f"  {r:22s} {a_:3d} {b_:3d}")


if __name__ == "__main__":
    main()
