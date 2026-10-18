"""Laptop screen whiteness per frame (no LLM): share of bright low-saturation pixels inside the COCO laptop box.
Spreadsheet/app ~ high; Windows lock screen (dark wallpaper) / off ~ low. -> data/screen/<video>.json {t: value or null}"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import frames
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent


def whiteness(im, box):
    _, x1, y1, x2, y2, _ = box
    a = np.asarray(im)
    H, W = a.shape[:2]
    c = a[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)]
    if c.size == 0:
        return None
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    return float(((hsv[..., 2] > 190) & (hsv[..., 1] < 50)).mean())


if __name__ == "__main__":
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    d = ROOT / "data/screen"; d.mkdir(exist_ok=True)
    for v in gt:
        ob = json.loads((ROOT / "data/objects" / f"{v}.json").read_text(encoding="utf-8"))
        out = {}
        for (t, im), boxes in zip(frames.sample(video_path(v), fps=1, max_frames=60, width=464, overlay=False), ob.values()):
            lb = [b for b in boxes if b[0] == 63]
            out[t] = round(whiteness(im, max(lb, key=lambda b: b[5])), 3) if lb else None
        (d / f"{v}.json").write_text(json.dumps(out), encoding="utf-8")
        vals = [x for x in out.values() if x is not None]
        print(f"{v:6s} {'V' if gt[v] else '.'} start {np.mean(vals[:3]) if vals else -1:.2f} end {np.mean(vals[-3:]) if vals else -1:.2f}", flush=True)
