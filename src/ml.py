"""Video-level violation classifier on aggregated per-frame features.

  python src/ml.py --models qwen4b gemma4b --clf lr
Sources per model m: runs/feat_<m>/preds.json (full-frame VLM answers), data/desk_<m>/ (desk-crop VLM answers).
Shared: data/people (YOLO persons), data/objects (YOLO COCO objects).
Evaluation: leave-one-out out-of-fold probabilities -> recall / FP / F2 at the 100%-recall threshold.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GT = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
VIDS = list(GT)
USE = {"desk": True, "feat": True, "coco": True, "yolo": True}
Y = np.array([1 if GT[v] else 0 for v in VIDS])
COCO = {24: "backpack", 26: "handbag", 28: "suitcase", 39: "bottle", 41: "cup", 63: "laptop", 67: "phone", 73: "book"}


def longest(x):
    best = cur = 0
    for b in x:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def agg(prefix, series):
    """series: dict key -> list of 0/1 (same length). Returns dict of aggregates."""
    f = {}
    for k, s in series.items():
        s = np.asarray(s, float)
        if not len(s):
            continue
        n = len(s)
        f[f"{prefix}{k}_mean"] = s.mean()
        f[f"{prefix}{k}_run"] = longest(s > 0) / n
        f[f"{prefix}{k}_end"] = s[-max(1, n // 6):].mean()
        f[f"{prefix}{k}_any"] = float(s.max() > 0)
    return f


def align(n, lst, fill=0):
    return (list(lst) + [lst[-1] if lst else fill] * n)[:n]


def video_features(v, models):
    F = {}
    ppl = json.loads((ROOT / "data/people" / f"{v}.json").read_text(encoding="utf-8"))
    n_in = [sum(1 for b in bs if b[3] > 0.40) for bs in ppl.values()]
    T = len(n_in)
    people = {"empty": [x == 0 for x in n_in], "two_plus": [x >= 2 for x in n_in], "three_plus": [x >= 3 for x in n_in]}
    F.update(agg("yolo_", people))
    F["yolo_people_max"] = max(n_in) if n_in else 0
    F["yolo_start_empty"] = float(len(n_in) > 1 and n_in[0] == 0)
    F["yolo_leaves"] = float(any(a >= 1 and b == 0 for a, b in zip(n_in, n_in[1:])))
    # number of separate client visits (rises from <=1 to >=2)
    F["yolo_visits"] = sum(1 for a, b in zip(n_in, n_in[1:]) if a <= 1 and b >= 2)
    of = ROOT / "data/objects" / f"{v}.json"
    if USE["coco"] and of.exists():
        ob = list(json.loads(of.read_text(encoding="utf-8")).values())
        ob = align(T, ob, [])
        F.update(agg("coco_", {name: [any(o[0] == c and o[4] > 0.35 for o in fr) for fr in ob] for c, name in COCO.items()}))
    for m in models:
        pf = ROOT / f"runs/feat_{m}/preds.json"
        if USE["feat"] and pf.exists():
            p = json.loads(pf.read_text(encoding="utf-8"))
            if v in p:
                raw = p[v]["raw"]
                keys = [k for k in raw[0] if k != "t"]
                S = {k: align(T, [r[k] for r in raw]) for k in keys}
                F.update(agg(f"{m}_f_", S))
                away = [e or not c for e, c in zip(people["empty"], S["cashier_in_room"])]
                for k in ["laptop_screen_on", "cash_on_desk", "safe_open", "cash_bag", "door_open"]:
                    F[f"{m}_x_away_{k}"] = longest([a and b for a, b in zip(away, S[k])]) / T
        dd = ROOT / f"data/desk_{m}" / f"{v}.json"
        if USE["desk"] and dd.exists():
            D = list(json.loads(dd.read_text(encoding="utf-8")).values())
            keys = list(D[0])
            S = {k: align(T, [d[k] for d in D]) for k in keys}
            F.update(agg(f"{m}_d_", S))
            fc = next((i for i, c in enumerate(S["cash"]) if c), None)
            F[f"{m}_x_id_before_cash"] = float(fc is not None and any(S["id_card"][:fc + 1]))
            F[f"{m}_x_cash_no_id"] = float(fc is not None and not any(S["id_card"]))
            F[f"{m}_x_client_no_id"] = float(max(n_in) >= 2 and not any(S["id_card"]))
    return F


FEATS = None  # optional regex whitelist


def matrix(models):
    import re
    rows = [video_features(v, models) for v in VIDS]
    cols = sorted(c for c in set().union(*rows) if c == "yolo_two_plus_run" or FEATS is None or re.search(FEATS, c))
    X = np.array([[r.get(c, 0.0) for c in cols] for r in rows], float)
    return X, cols


class Cover:
    """Greedy OR-of-thresholds: pick (feature >= thr) rules until all training positives are covered,
    each step maximizing new positives - lam * new false positives. Predict score = 1 if any rule fires."""
    def __init__(self, lam=1.5, max_rules=8):
        self.lam, self.max_rules = lam, max_rules

    def fit(self, X, y):
        cand = []
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                if t > 0:
                    cand.append((j, t, X[:, j] >= t))
        covered = np.zeros(len(y), bool); flagged = np.zeros(len(y), bool); self.rules = []
        while not covered[y == 1].all() and len(self.rules) < self.max_rules:
            best = max(cand, key=lambda c: (((c[2] & ~covered) & (y == 1)).sum()
                                            - self.lam * ((c[2] & ~flagged) & (y == 0)).sum(), -c[2][y == 0].sum()))
            gain = ((best[2] & ~covered) & (y == 1)).sum()
            if gain == 0:
                break
            self.rules.append(best[:2]); covered |= best[2] & (y == 1); flagged |= best[2]
        return self

    def predict_proba(self, X):
        f = np.zeros(len(X))
        for j, t in self.rules:
            f = np.maximum(f, (X[:, j] >= t).astype(float))
        return np.c_[1 - f, f]


def make_clf(name):
    if name.startswith("cover"):
        return Cover(lam=float(name[5:] or 1.5))
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    if name == "rf":
        return RandomForestClassifier(300, max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=0)
    if name == "et":
        return ExtraTreesClassifier(500, max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=0)
    raise ValueError(name)


def oof(X, clf_name):
    p = np.zeros(len(Y))
    for i in range(len(Y)):
        tr = np.arange(len(Y)) != i
        c = make_clf(clf_name).fit(X[tr], Y[tr])
        p[i] = c.predict_proba(X[i:i + 1])[0, 1]
    return p


def report(p, label=""):
    thr = p[Y == 1].min()  # lowest violation score -> 100% recall
    pred = p >= thr
    tp = int((pred & (Y == 1)).sum()); fp = int((pred & (Y == 0)).sum())
    prec = tp / max(1, tp + fp); rec = tp / Y.sum()
    f2 = 5 * prec * rec / (4 * prec + rec)
    # best-F2 operating point (for reference)
    best = max(((5 * ((p >= t) & (Y == 1)).sum() / max(1, (p >= t).sum()) * ((p >= t) & (Y == 1)).sum() / Y.sum())
                / max(1e-9, 4 * ((p >= t) & (Y == 1)).sum() / max(1, (p >= t).sum()) + ((p >= t) & (Y == 1)).sum() / Y.sum()), t)
               for t in np.unique(p))
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(Y, p)
    b = p >= 0.5; tp5 = int((b & (Y == 1)).sum()); fp5 = int((b & (Y == 0)).sum())
    pr5 = tp5 / max(1, tp5 + fp5); rc5 = tp5 / Y.sum(); f25 = 5 * pr5 * rc5 / max(1e-9, 4 * pr5 + rc5)
    print(f"{label:40s} @0.5: recall {tp5}/{Y.sum()} FP {fp5}/{(Y == 0).sum()} F2 {f25:.3f} | AUC {auc:.3f} | @recall=1: FP {fp}/{(Y == 0).sum()} prec {prec:.2f} F2 {f2:.3f} | best F2 {best[0]:.3f}")
    return dict(auc=auc, fp=fp, f2=f2, best_f2=best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen4b"])
    ap.add_argument("--clf", nargs="+", default=["lr", "rf", "et"])
    ap.add_argument("--off", nargs="*", default=[], help="disable sources: desk feat coco")
    ap.add_argument("--top", type=int, default=0, help="print top-N features by single-feature AUC")
    a = ap.parse_args()
    for k in a.off:
        USE[k] = False
    X, cols = matrix(a.models)
    print(f"videos {len(Y)} (viol {Y.sum()}), features {len(cols)}; always-yes F2 = {5*Y.mean()/(4*Y.mean()+1):.3f}")
    if a.top:
        from sklearn.metrics import roc_auc_score
        sc = sorted(((roc_auc_score(Y, X[:, j]), c) for j, c in enumerate(cols) if X[:, j].std() > 0), reverse=True)
        for s, c in sc[:a.top]:
            print(f"  {s:.3f} {c}")
    out = {}
    for c in a.clf:
        p = oof(X, c)
        out[c] = report(p, f"{'+'.join(a.models)} {c}")
        np.save(ROOT / f"runs/oof_{'+'.join(a.models)}_{c}.npy", p)
    return out


if __name__ == "__main__":
    main()
