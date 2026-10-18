"""End-to-end detector for one video (no caches): features -> events -> jury JSON.
  python src/detect.py path/to/video.mov [--out result.json]
Models are loaded one at a time (YOLO on CPU, Qwen3-VL-4B, then Gemma-3-4B) to fit 12 GB VRAM.
"""
import argparse, base64, json, re, sys, time
from pathlib import Path

import cv2, numpy as np, torch
from torch.nn.attention import sdpa_kernel

sys.path.insert(0, str(Path(__file__).parent))
import frames, vlm, state, events
from vlm import backends

ROOT = Path(__file__).resolve().parent.parent
QWEN, GEMMA = "models/Qwen3-VL-4B-Instruct", "models/gemma-3-4b-it"
RULES = {r["id"]: r for r in json.loads((ROOT / "data/rules.json").read_text(encoding="utf-8"))}
_yolo = None


def yolo(fr):
    global _yolo
    from ultralytics import YOLO
    _yolo = _yolo or YOLO("yolo11m.pt")
    ppl, obj = {}, {}
    for t, im in fr:
        r = _yolo(np.asarray(im)[:, :, ::-1], classes=[0, 63], conf=0.25, device="cpu", verbose=False)[0]
        h, w = r.orig_shape
        bx = [[int(c), x1 / w, y1 / h, x2 / w, y2 / h, p] for (x1, y1, x2, y2), c, p in
              zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        ppl[t] = [b[1:] for b in bx if b[0] == 0 and b[5] >= 0.35]
        obj[t] = bx
    return ppl, obj


def ground(im):
    import ground as G
    G.MODEL = QWEN
    return {k: G.locate(im, q) for k, q in G.OBJ.items()}


def desk_crops(fr, g):
    if g:
        h = g[3] - g[1]
        B = (max(0, g[0] - 0.05), max(0, g[1] - 1.2 * h), min(1, g[2] + 0.05), min(1, g[3] + 0.1 * h))
    else:
        B = (0.0, 0.30, 1.0, 0.62)
    out = []
    for _, im in fr:
        W, H = im.size
        c = im.crop((int(B[0] * W), int(B[1] * H), int(B[2] * W), int(B[3] * H)))
        out.append(c.resize((c.width * 2, c.height * 2)))
    return out


def mc(model_path, images, question, letters="ABC"):
    import safe as S
    model, proc = vlm.load(model_path, "none"); proc.tokenizer.padding_side = "left"
    res = []
    for i in range(0, len(images), 8):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": c}, {"type": "text", "text": question}]}] for c in images[i:i + 8]]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode(), sdpa_kernel(backends(model)):
            out = model.generate(**inp, max_new_tokens=3, do_sample=False)
        for o in out:
            m = re.search(f"[{letters}]", proc.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True))
            res.append(m.group(0) if m else "?")
    return res


def screen_feats(fr, obj):
    S = []
    for (t, im), bx in zip(fr, obj.values()):
        lb = [b for b in bx if b[0] == 63]
        if lb:
            _, x1, y1, x2, y2, _ = max(lb, key=lambda b: b[5]); a = np.asarray(im).astype(float); H, W = a.shape[:2]
            S.append((a[int(y1 * H):int((y1 + 0.6 * (y2 - y1)) * H), int(x1 * W):int(x2 * W)].mean(), a.mean()))
    if len(S) < 4:
        return None, 0.0
    s0 = np.mean([x[0] for x in S[:3]]); s1 = np.mean([x[0] for x in S[-3:]])
    f0 = np.mean([x[1] for x in S[:3]]); f1 = np.mean([x[1] for x in S[-3:]])
    return float((s1 / s0) / (f1 / f0)), float(np.mean([x[0] for x in S[:6]]))


def residual(fr, obj):
    import residual as R
    g = lambda im: cv2.GaussianBlur(cv2.cvtColor(np.asarray(im.resize((232, int(232 * im.height / im.width)))), cv2.COLOR_RGB2GRAY).astype(float), (5, 5), 0)
    a = np.mean([g(im) for _, im in fr[:2]], 0); b = np.mean([g(im) for _, im in fr[-2:]], 0)
    a = a / a.mean() * 128; b = b / b.mean() * 128
    ob = list(obj.values())
    boxes = [x for i in (0, 1, -2, -1) for x in ob[i] if x[0] in (0, 63)]
    m = R.mask_boxes(a.shape, boxes)
    return float((np.abs(a - b)[m] > 25).mean())


def analyze(path):
    t0 = time.time(); timing = {}
    fr = frames.sample(path, fps=1, max_frames=60, width=464, overlay=False)
    ppl, obj = yolo(fr); timing["yolo"] = time.time() - t0
    g = ground(fr[len(fr) // 2][1])
    dc = desk_crops(fr, g["desk"])
    Q = state.ask_frames({"model": QWEN, "quant": "none", "batch": 4, "state_tokens": 90,
                          "state_system": "desk_system_v1.txt", "state_prompt": "desk_v1.txt"}, dc)
    import safe as S
    sf = {}
    if g["safe"]:
        sf = dict(zip([str(t) for t, _ in fr], mc(QWEN, [S.crop(im, g["safe"]) for _, im in fr], S.Q)))
    timing["qwen"] = time.time() - t0
    D = state.ask_frames({"model": GEMMA, "quant": "none", "batch": 6, "state_tokens": 90,
                          "state_system": "desk_system_v1.txt", "state_prompt": "desk_v1.txt"}, dc)
    timing["gemma"] = time.time() - t0
    sr, on = screen_feats(fr, obj)
    F = dict(ppl={str(t): v for t, v in ppl.items()}, D=D, Q=Q, sr=sr, rs=residual(fr, obj), on=on, sf=sf)
    ev = events.events(F=F)
    viol = []
    for rid, t, desc in ev:
        r = RULES.get(rid, {})
        jpg = frames.frame_at(path, t)
        viol.append({"type": r.get("title", rid), "rule_id": rid, "source": r.get("source", ""),
                     "timecode_sec": round(float(t), 1), "description": desc,
                     "evidence_frames": [{"timestamp_sec": round(float(t), 1),
                                          "image_base64": base64.b64encode(jpg).decode() if jpg else ""}]})
    timing["total"] = time.time() - t0
    return {"status": "done", "verdict": "violation" if viol else "no_violation", "violations": viol,
            "debug": {"timing_sec": {k: round(v, 1) for k, v in timing.items()}, "ground": g}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--out")
    a = ap.parse_args()
    res = analyze(a.video)
    short = {**res, "violations": [{k: v for k, v in x.items() if k != "evidence_frames"} for x in res["violations"]]}
    print(json.dumps(short, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
