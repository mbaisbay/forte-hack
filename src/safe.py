"""Safe door state on grounded safe crop (VLM multiple choice), every frame. -> data/safe_<tag>/<video>.json {t: A|B|C}"""
import argparse, json, re, sys
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel
sys.path.insert(0, str(Path(__file__).parent))
import frames, vlm
from vlm import backends
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
Q = """This is a zoomed crop around a grey metal safe in a bank cashier room.
Look at the FRONT DOOR of the safe. Choose one:
A) the safe door is open (swung outward, or the dark inside of the safe is visible)
B) the safe door is closed (flat front, flush with the body)
C) the safe is hidden / not visible
Answer with one letter only."""


def crop(im, g):
    W, H = im.size; pw = (g[2] - g[0]) * 0.35; ph = (g[3] - g[1]) * 0.2
    c = im.crop((int(max(0, g[0] - pw) * W), int(max(0, g[1] - ph) * H), int(min(1, g[2] + pw * 0.3) * W), int(min(1, g[3] + ph) * H)))
    s = 448 / max(c.size)
    return c.resize((int(c.width * s), int(c.height * s)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="*")
    ap.add_argument("--model", default="models/gemma-3-4b-it"); ap.add_argument("--tag", default="gemma4b"); ap.add_argument("--quant", default="none")
    a = ap.parse_args()
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    d = ROOT / f"data/safe_{a.tag}"; d.mkdir(parents=True, exist_ok=True)
    model, proc = vlm.load(a.model, a.quant); proc.tokenizer.padding_side = "left"
    for v in (a.videos or gt):
        f = d / f"{v}.json"
        g = json.loads((ROOT / "data/ground" / f"{v}.json").read_text(encoding="utf-8"))["safe"]
        if f.exists() or not g:
            continue
        fr = frames.sample(video_path(v), fps=1, max_frames=60, width=464, overlay=False)
        ims = [(t, crop(im, g)) for t, im in fr]; res = {}
        for i in range(0, len(ims), 8):
            msgs = [[{"role": "user", "content": [{"type": "image", "image": c}, {"type": "text", "text": Q}]}] for _, c in ims[i:i + 8]]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode(), sdpa_kernel(backends(model)):
                out = model.generate(**inp, max_new_tokens=3, do_sample=False)
            for (t, _), o in zip(ims[i:i + 8], out):
                m = re.search(r"[ABC]", proc.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True))
                res[str(t)] = m.group(0) if m else "?"
        f.write_text(json.dumps(res), encoding="utf-8")
        print(v, "V" if gt[v] else ".", "".join(res.values()), flush=True)
