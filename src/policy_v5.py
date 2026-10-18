"""Rule policy v6 (v5 + safe open at end, Qwen4B on grounded safe crop) with leave-one-out threshold selection (honest estimate of the hand-built policy).

flag = client_session OR personal_items OR door_left_open OR (cashier_left AND laptop_not_locked)
  client_session : YOLO >=2 people inside room for >= T_client of frames
  personal_items : Gemma4B desk-crop personal_item longest run >= T_pers
  door_left_open : end-vs-start residual change (outside people/laptop) in (T_door, 0.5)
  cashier_left   : YOLO room empty at the end
  laptop locked  : screen brightness end/start normalized by frame < T_lock
Thresholds: for each held-out video, grid-search on the other 39 -> max recall, then min FP, then max margin.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ml

ROOT = Path(__file__).resolve().parent.parent
GRID = dict(T_client=[0.2, 0.3, 0.4, 0.5], T_pers=[0.25, 0.35, 0.45, 0.55, 0.65], T_door=[0.015, 0.02, 0.03, 0.04],
            T_lock=[0.7, 0.75, 0.8, 0.85])


def signals(model="gemma4b"):
    sr = json.loads((ROOT / "data/screen_ratio.json").read_text(encoding="utf-8"))
    rs = json.loads((ROOT / "data/residual.json").read_text(encoding="utf-8"))
    X, cols = ml.matrix([model])
    c = lambda k: X[:, cols.index(k)]
    safe = []
    for v in ml.VIDS:
        f = ROOT / "data/safe_qwen4b" / f"{v}.json"
        st = list(json.loads(f.read_text(encoding="utf-8")).values()) if f.exists() else []
        safe.append(sum(x == "A" for x in st[-3:]) >= 2)
    bag = json.loads((ROOT / ("data/bag_count_g.json" if model.endswith("g") else "data/bag_count.json")).read_text(encoding="utf-8"))
    on = json.loads((ROOT / "data/screen_on.json").read_text(encoding="utf-8"))
    return dict(bag=np.array([bag[v] for v in ml.VIDS]), scr_on=np.array([on[v] for v in ml.VIDS]), safe_end=np.array(safe), client=c("yolo_two_plus_run"), pers=c(f"{model}_d_personal_item_run"), left=c("yolo_empty_end") > 0,
                res=np.array([rs[v] for v in ml.VIDS]),
                lock=np.array([sr[v] if sr[v] is not None else 9.0 for v in ml.VIDS]))


VARIANT = "v6"


def predict(S, T_client, T_pers, T_door, T_lock):
    if VARIANT == "v7":  # left-room net only when screen was on; bag rule for screen-off cases
        return ((S["client"] >= T_client) | (S["pers"] >= T_pers) | ((S["res"] > T_door) & (S["res"] < 0.5))
                | (S["left"] & (S["scr_on"] >= 120) & ~(S["lock"] < T_lock)) | (S["left"] & (S["bag"] >= 3)) | S["safe_end"])
    return ((S["client"] >= T_client) | (S["pers"] >= T_pers) | ((S["res"] > T_door) & (S["res"] < 0.5))
            | (S["left"] & ~(S["lock"] < T_lock)) | S["safe_end"])


def score(pred, y):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    pr = tp / max(1, pred.sum()); rc = tp / max(1, (y == 1).sum())
    return tp, fp, (5 * pr * rc / (4 * pr + rc) if pr + rc else 0.0)


def main(model="gemma4b"):
    S, Y = signals(model), ml.Y
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    full = max(combos, key=lambda p: (score(predict(S, **p), Y)[0], -score(predict(S, **p), Y)[1]))
    print("in-sample best:", full, score(predict(S, **full), Y))
    pred = np.zeros(len(Y), bool)
    for i in range(len(Y)):
        m = np.arange(len(Y)) != i
        Si = {k: v[m] for k, v in S.items()}
        best = max(combos, key=lambda p: (score(predict(Si, **p), Y[m])[0], -score(predict(Si, **p), Y[m])[1]))
        pred[i] = predict({k: v[i:i + 1] for k, v in S.items()}, **best)[0]
    tp, fp, f2 = score(pred, Y)
    print(f"LOO: recall {tp}/{Y.sum()} FP {fp}/{(Y == 0).sum()} F2 {f2:.3f}  missed",
          [v for v, p, y in zip(ml.VIDS, pred, Y) if y and not p], "FP", [v for v, p, y in zip(ml.VIDS, pred, Y) if p and not y])


if __name__ == "__main__":
    if len(sys.argv) > 2:
        VARIANT = sys.argv[2]
    main(sys.argv[1] if len(sys.argv) > 1 else "gemma4b")
