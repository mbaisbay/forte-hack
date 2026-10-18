"""Desk-crop per-frame features (VLM on native-res desk crop, upscaled 2x). Cached to data/desk/<video>.json."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import frames, state

ROOT = Path(__file__).resolve().parent.parent
CFG = {"model": "models/Qwen3-VL-4B-Instruct", "quant": "none", "batch": 4, "state_tokens": 90,
       "state_system": "desk_system_v1.txt", "state_prompt": "desk_v1.txt"}
BOX = (0.0, 0.30, 1.0, 0.62)  # desk region (x1,y1,x2,y2 normalized) for the training camera


def desk_box(video):
    """Grounded desk box (Qwen3-VL, data/ground) padded upward for hands/handover; fallback: fixed band."""
    f = ROOT / "data/ground" / f"{Path(video).stem.replace('X', 'Х')}.json"
    g = json.loads(f.read_text(encoding="utf-8")).get("desk") if f.exists() else None
    if not g or not GROUNDED:
        return BOX
    h = g[3] - g[1]
    return (max(0, g[0] - 0.05), max(0, g[1] - 1.2 * h), min(1, g[2] + 0.05), min(1, g[3] + 0.1 * h))


GROUNDED = False


def crops(video, fps=1, max_frames=60):
    fr = frames.sample(video, fps=fps, max_frames=max_frames, width=464, overlay=False)
    out = []
    B = desk_box(video)
    for t, im in fr:
        w, h = im.size
        c = im.crop((int(B[0] * w), int(B[1] * h), int(B[2] * w), int(B[3] * h)))
        out.append((t, c.resize((c.width * 2, c.height * 2))))
    return out


if __name__ == "__main__":
    from evaluate import video_path
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="*")
    ap.add_argument("--model", default=CFG["model"]); ap.add_argument("--quant", default="none")
    ap.add_argument("--out", default="data/desk")
    ap.add_argument("--grounded", action="store_true")
    a = ap.parse_args()
    CFG.update(model=a.model, quant=a.quant)
    GROUNDED = a.grounded
    d = ROOT / a.out; d.mkdir(parents=True, exist_ok=True)
    for v in (a.videos or gt):
        f = d / f"{v}.json"
        if f.exists():
            continue
        cr = crops(video_path(v))
        S = state.ask_frames(CFG, [im for _, im in cr])
        f.write_text(json.dumps({str(t): s for (t, _), s in zip(cr, S)}), encoding="utf-8")
        print(v, "".join(str(s["id_card"]) for s in S), flush=True)
