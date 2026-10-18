"""YOLO (COCO) detections per sampled frame -> data/objects/<video>.json {t: [[cls,x1,y1,x2,y2,conf],...]} normalized."""
import json, sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).parent))
import frames
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
CLS = [0, 24, 26, 28, 39, 41, 62, 63, 64, 66, 67, 73]  # person backpack handbag suitcase bottle cup tv laptop mouse keyboard phone book

if __name__ == "__main__":
    m = YOLO("yolo11m.pt")
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    d = ROOT / "data/objects"; d.mkdir(exist_ok=True)
    for v in gt:
        f = d / f"{v}.json"
        if f.exists():
            continue
        out = {}
        for t, im in frames.sample(video_path(v), fps=1, max_frames=60, width=464, overlay=False):
            r = m(np.asarray(im)[:, :, ::-1], classes=CLS, conf=0.25, device="cpu", verbose=False)[0]
            h, w = r.orig_shape
            out[t] = [[int(c), round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / h, 3), round(p, 2)]
                      for (x1, y1, x2, y2), c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        f.write_text(json.dumps(out), encoding="utf-8")
        print(v, flush=True)
