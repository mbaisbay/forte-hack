"""Two-stage policy: client session (YOLO >=2 people for >=30% frames) -> flag; solo videos -> learned detector (LOO on solo subset).
  python src/policy.py --models qwen4b --clf cover1 et lr
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ml


def run(models, clf, off=(), client_thr=0.3, verbose=False):
    for k in off:
        ml.USE[k] = False
    X, cols = ml.matrix(models)
    Y = ml.Y
    client = X[:, cols.index("yolo_two_plus_run")] >= client_thr
    solo = np.where(~client)[0]
    pred = client.astype(float)
    Xs, Ys = X[solo], Y[solo]
    for k, i in enumerate(solo):
        tr = np.arange(len(solo)) != k
        c = ml.make_clf(clf).fit(Xs[tr], Ys[tr])
        p = c.predict_proba(Xs[k:k + 1])[0, 1]
        if clf.startswith("cover"):
            pred[i] = p
        else:  # threshold = lowest training positive score (in-sample) -> aim at recall 1
            ptr = c.predict_proba(Xs[tr])[:, 1]
            pred[i] = float(p >= ptr[Ys[tr] == 1].min())
        if verbose and pred[i] != Y[i]:
            print("   err", ml.VIDS[i], "y", Y[i], "rules", getattr(c, "rules", "")[:3] and [(cols[j], round(t, 2)) for j, t in c.rules])
    b = pred >= 0.5
    tp = int((b & (Y == 1)).sum()); fp = int((b & (Y == 0)).sum())
    pr = tp / max(1, tp + fp); rc = tp / Y.sum(); f2 = 5 * pr * rc / max(1e-9, 4 * pr + rc)
    solo_fp = int((b[solo] & (Ys == 0)).sum()); solo_fn = int((~b[solo] & (Ys == 1)).sum())
    print(f"{'+'.join(models):24s} {clf:7s} recall {tp}/{Y.sum()}  FP {fp}/{(Y == 0).sum()}  F2 {f2:.3f}  "
          f"(solo: FN {solo_fn}/{Ys.sum()} FP {solo_fp}/{(Ys == 0).sum()})")
    return f2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen4b"])
    ap.add_argument("--clf", nargs="+", default=["cover1", "cover2", "et", "lr"])
    ap.add_argument("--off", nargs="*", default=[])
    ap.add_argument("--feats", default=None)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    ml.FEATS = a.feats
    for c in a.clf:
        run(a.models, c, a.off, verbose=a.v)
