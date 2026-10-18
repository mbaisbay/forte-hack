"""Simulate another camera angle: horizontal flip + 4 deg rotation + off-center 85% crop. -> data/aug/aug_<i>.mov + map.json"""
import json, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).parent))
from frames import _open
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    d = ROOT / "data/aug"; d.mkdir(exist_ok=True)
    mapping = {}
    for i, v in enumerate(gt):
        name = d / f"aug_{i}.mov"
        mapping[name.name] = v
        if name.exists():
            continue
        cap = _open(video_path(v)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out = None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.flip(f, 1); H, W = f.shape[:2]
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 4, 1.0)
            f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            f = f[int(0.05 * H):int(0.90 * H), int(0.10 * W):int(0.95 * W)]
            if out is None:
                out = cv2.VideoWriter(str(name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
            out.write(f)
        out.release()
        print(v, name.name, flush=True)
    (d / "map.json").write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
