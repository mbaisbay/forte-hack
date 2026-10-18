"""End-vs-start scene change outside people/laptop boxes (no model). -> data/residual.json {video: value}"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import frames
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent


def mask_boxes(shape, boxes, pad=0.03):
    m = np.ones(shape[:2], bool); H, W = shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = b[1:5]
        m[int(max(0, y1 - pad) * H):int(min(1, y2 + pad) * H), int(max(0, x1 - pad) * W):int(min(1, x2 + pad) * W)] = False
    return m


if __name__ == "__main__":
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    res = {}
    for v in gt:
        fr = frames.sample(video_path(v), fps=1, max_frames=60, width=232, overlay=False)
        ob = list(json.loads((ROOT / "data/objects" / f"{v}.json").read_text(encoding="utf-8")).values())
        g = lambda im: cv2.GaussianBlur(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY).astype(float), (5, 5), 0)
        a = np.mean([g(im) for _, im in fr[:2]], 0); b = np.mean([g(im) for _, im in fr[-2:]], 0)
        a = a / a.mean() * 128; b = b / b.mean() * 128  # exposure normalize
        boxes = [x for i in (0, 1, -2, -1) for x in ob[i] if x[0] in (0, 63)]
        m = mask_boxes(a.shape, boxes)
        res[v] = float((np.abs(a - b)[m] > 25).mean())
        print(f"{v:6s} {'V ' + gt[v][0]['rules'][0] if gt[v] else 'clean':11s} residual {res[v]:.3f}", flush=True)
    (ROOT / "data/residual.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
