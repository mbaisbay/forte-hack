"""Video -> violations. Two modes: 'all' (one prompt, all rules) and 'per_rule' (one yes/no prompt per rule)."""
import json, re, time
from pathlib import Path

import frames, vlm

ROOT = Path(__file__).resolve().parent.parent


def load_rules(cfg):
    rules = json.loads((ROOT / cfg.get("rules", "data/rules.json")).read_text(encoding="utf-8"))
    return [r for r in rules if r["id"] not in cfg.get("disabled_rules", [])]


def rule_block(r, fewshot, exclude=()):
    s = f"[{r['id']}] {r['title']} ({r['source']})\nRule: {r['text']}\nLook for: {r['check']}"
    ex = [e for e in r.get("fewshot", []) if not any(x + " " in e for x in exclude)]
    if fewshot and ex:
        s += "\nExamples:\n" + "\n".join("- " + e for e in ex)
    return s


def parse_json(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    m = re.search(r"\{.*\}|\[.*\]", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def build_content(fr, cfg):
    content = []
    for t, im in fr:
        if cfg.get("text_timestamps", True):
            content.append(f"t={t:.1f}s")
        content.append(im)
    return content


def analyze(video, cfg, exclude_fewshot=()):
    """exclude_fewshot: video ids whose few-shot examples must be hidden (to avoid leakage when evaluating)."""
    P = lambda name: (ROOT / "prompts" / cfg[name]).read_text(encoding="utf-8")
    t0 = time.time()
    fr = frames.sample(video, fps=cfg["fps"], max_frames=cfg["max_frames"], width=cfg["width"],
                       overlay=cfg.get("overlay", True))
    dur = fr[-1][0] if fr else 0
    content = build_content(fr, cfg) if cfg["mode"] != "state" else []
    rules = load_rules(cfg)
    m, q = cfg["model"], cfg.get("quant", "none")
    viols, raw, ntok = [], [], 0
    if cfg["mode"] == "state":
        import state
        S = state.ask_frames(cfg, [im for _, im in fr])
        ts = [t for t, _ in fr]
        raw = [{"t": t, **s} for t, s in zip(ts, S)]
        for rid, t, desc in state.apply_rules(ts, S, cfg["thresholds"]):
            viols.append({"rule_id": rid, "timecode_sec": t, "description": desc})
    elif cfg["mode"] == "all":
        catalog = "\n\n".join(rule_block(r, cfg["fewshot"], exclude_fewshot) for r in rules)
        prompt = P("prompt_all").replace("{RULES}", catalog).replace("{DURATION}", f"{dur:.0f}")
        out, ntok = vlm.generate(m, q, P("system"), content + [prompt], cfg.get("max_new_tokens", 768))
        raw.append(out)
        js = parse_json(out) or {}
        items = js.get("violations", []) if isinstance(js, dict) else js
        for v in items or []:
            if isinstance(v, dict) and v.get("rule_id"):
                viols.append(v)
    else:
        for r in rules:
            prompt = P("prompt_rule").replace("{RULE}", rule_block(r, cfg["fewshot"], exclude_fewshot)) \
                .replace("{DURATION}", f"{dur:.0f}")
            out, ntok = vlm.generate(m, q, P("system"), content + [prompt], cfg.get("max_new_tokens", 256))
            raw.append(r["id"] + ": " + out)
            js = parse_json(out)
            if isinstance(js, dict) and js.get("violation") in (True, "true", "yes") \
                    and float(js.get("confidence", 1)) >= cfg.get("min_conf", 0):
                js["rule_id"] = r["id"]
                viols.append(js)
    byid = {r["id"]: r for r in rules}
    res = []
    for v in viols:
        try:
            t = float(v.get("timecode_sec", 0))
        except (TypeError, ValueError):
            t = 0.0
        r = byid.get(str(v["rule_id"]).strip("[] "), {})
        res.append({"rule_id": str(v["rule_id"]).strip("[] "), "type": r.get("title", str(v["rule_id"])),
                    "source": r.get("source", ""), "timecode_sec": round(min(max(t, 0), dur), 1),
                    "description": v.get("description", ""), "confidence": v.get("confidence")})
    res = dedup(res, cfg.get("dedup_sec", 3))
    return {"violations": res, "raw": raw, "sec": round(time.time() - t0, 1), "tokens": ntok, "duration": dur}


def dedup(vs, tol):
    out = []
    for v in sorted(vs, key=lambda v: v["timecode_sec"]):
        if any(o["rule_id"] == v["rule_id"] and abs(o["timecode_sec"] - v["timecode_sec"]) <= tol for o in out):
            continue
        out.append(v)
    return out
