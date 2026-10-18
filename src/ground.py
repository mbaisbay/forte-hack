"""Open-vocabulary localization with Qwen3-VL grounding: one frame per video -> boxes for safe / door / desk.
-> data/ground/<video>.json {"safe": [x1,y1,x2,y2] normalized or null, ...}
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import frames, vlm
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "models/Qwen3-VL-4B-Instruct"
OBJ = {"safe": "the grey metal safe (a box-shaped metal cabinet for cash)", "door": "the glass door of the room",
       "desk": "the cashier's desk"}


def locate(im, what):
    out, _ = vlm.generate(MODEL, "none", "You are a precise visual grounding assistant.",
                          [im, f"Locate {what} in the image. Output JSON: {{\"bbox_2d\": [x1, y1, x2, y2]}}"], 60)
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", out)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return [round(x1 / 1000, 3), round(y1 / 1000, 3), round(x2 / 1000, 3), round(y2 / 1000, 3)]  # Qwen3-VL uses 0-1000 coords


if __name__ == "__main__":
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    d = ROOT / "data/ground"; d.mkdir(exist_ok=True)
    for v in (sys.argv[1:] or gt):
        f = d / f"{v}.json"
        if f.exists():
            continue
        fr = frames.sample(video_path(v), fps=1, max_frames=60, width=464, overlay=False)
        im = fr[len(fr) // 2][1]
        res = {k: locate(im, q) for k, q in OBJ.items()}
        f.write_text(json.dumps(res), encoding="utf-8")
        print(v, res, flush=True)
