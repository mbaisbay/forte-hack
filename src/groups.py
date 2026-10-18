"""Per violation-group separability: best single features (AUC group-positives vs all clean videos)."""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
import ml

GROUPS = {"ID 231.19.1": ["231.19.1"], "cash/safe left 231.20/22": ["231.20.1", "231.22.1", "231.21.1"],
          "laptop 53.26": ["53.26.1"], "personal 231.12": ["231.12.1"], "door/outsider 14.08/75/14.21": ["14.08.1", "231.75.1", "14.21.1"],
          "receipt 231.19.6": ["231.19.6"], "other (19.4,24.3,46)": ["231.19.4", "231.24.3", "231.46.1", "231.43.1"]}

if __name__ == "__main__":
    models = sys.argv[1:] or ["qwen4b"]
    X, cols = ml.matrix(models)
    clean = ml.Y == 0
    for g, rules in GROUPS.items():
        pos = np.array([any(r in e["rules"] for e in ml.GT[v] for r in rules) for v in ml.VIDS])
        m = pos | clean
        sc = sorted(((roc_auc_score(pos[m], X[m, j]), c) for j, c in enumerate(cols) if X[m, j].std() > 0), reverse=True)[:3]
        print(f"{g:30s} n={pos.sum():2d}  " + " | ".join(f"{c} {s:.2f}" for s, c in sc))
