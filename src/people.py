"""YOLO person boxes per sampled frame, cached to data/people/<video>.json as {t: [[x1,y1,x2,y2,conf],...]} (normalized)."""
import json, sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
import frames

ROOT = Path(__file__).resolve().parent.parent
_model = None


def detect(video, fps=1, max_frames=60):
    global _model
    _model = _model or YOLO("yolo11s.pt")
    fr = frames.sample(video, fps=fps, max_frames=max_frames, width=464, overlay=False)
    out = {}
    for t, im in fr:
        r = _model(np.asarray(im)[:, :, ::-1], classes=[0], conf=0.35, device="cpu", verbose=False)[0]
        h, w = r.orig_shape
        out[t] = [[round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / h, 3), round(c, 2)]
                  for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    return out


if __name__ == "__main__":
    from evaluate import video_path
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    d = ROOT / "data/people"; d.mkdir(exist_ok=True)
    for v in gt:
        f = d / f"{v}.json"
        if not f.exists():
            f.write_text(json.dumps(detect(video_path(v))), encoding="utf-8")
            print(v, flush=True)
