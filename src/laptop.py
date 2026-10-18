"""Laptop-screen state per frame: crop laptop box (COCO 'laptop' from data/objects) -> VLM multiple choice.
Cached to data/laptop_<m>/<video>.json {t: "A"|"B"|"C"|"D"|"-"} ('-' = no laptop box)."""
import argparse, json, re, sys
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel
sys.path.insert(0, str(Path(__file__).parent))
import frames, vlm
from vlm import backends
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
Q = (ROOT / "prompts/laptop_v1.txt").read_text(encoding="utf-8")


def crops(v):
    ob = json.loads((ROOT / "data/objects" / f"{v}.json").read_text(encoding="utf-8"))
    out = []
    for (t, im), boxes in zip(frames.sample(video_path(v), fps=1, max_frames=60, width=464, overlay=False), ob.values()):
        lb = [b for b in boxes if b[0] == 63]
        if not lb:
            out.append((t, None)); continue
        _, x1, y1, x2, y2, _ = max(lb, key=lambda b: b[5])
        W, H = im.size; pw, ph = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        c = im.crop((int(max(0, x1 - pw) * W), int(max(0, y1 - ph) * H), int(min(1, x2 + pw) * W), int(min(1, y2 + ph) * H)))
        s = 448 / max(c.size)
        out.append((t, c.resize((int(c.width * s), int(c.height * s)))))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="*")
    ap.add_argument("--model", default="models/gemma-3-4b-it"); ap.add_argument("--quant", default="none")
    ap.add_argument("--out", default="data/laptop_gemma4b")
    a = ap.parse_args()
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    d = ROOT / a.out; d.mkdir(parents=True, exist_ok=True)
    model, proc = vlm.load(a.model, a.quant)
    proc.tokenizer.padding_side = "left"
    for v in (a.videos or gt):
        f = d / f"{v}.json"
        if f.exists():
            continue
        cr = crops(v); res = {}
        ims = [(t, c) for t, c in cr if c is not None]
        for i in range(0, len(ims), 8):
            msgs = [[{"role": "user", "content": [{"type": "image", "image": c}, {"type": "text", "text": Q}]}] for _, c in ims[i:i + 8]]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode(), sdpa_kernel(backends(model)):
                g = model.generate(**inp, max_new_tokens=4, do_sample=False)
            for (t, _), o in zip(ims[i:i + 8], g):
                m = re.search(r"[ABCD]", proc.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True))
                res[str(t)] = m.group(0) if m else "?"
        for t, c in cr:
            res.setdefault(str(t), "-")
        res = dict(sorted(res.items(), key=lambda kv: float(kv[0])))
        f.write_text(json.dumps(res), encoding="utf-8")
        print(v, "".join(res.values()), flush=True)
