"""Per-frame state extraction (VLM answers fixed yes/no questions per frame) + rule engine in code."""
import json, re
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

import vlm
from vlm import backends

ROOT = Path(__file__).resolve().parent.parent


def ask_frames(cfg, images):
    """Batched: one frame per sequence. Returns list of dicts {key: 0/1}."""
    model, proc = vlm.load(cfg["model"], cfg.get("quant", "none"))
    proc.tokenizer.padding_side = "left"
    sysmsg = (ROOT / "prompts" / cfg["state_system"]).read_text(encoding="utf-8")
    q = (ROOT / "prompts" / cfg["state_prompt"]).read_text(encoding="utf-8")
    keys = re.findall(r"^(\w+):", q, re.M)
    out = []
    bs = cfg.get("batch", 6)
    for i in range(0, len(images), bs):
        msgs = [[{"role": "system", "content": [{"type": "text", "text": sysmsg}]},
                 {"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": q}]}]
                for im in images[i:i + bs]]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode(), sdpa_kernel(backends(model)):
            gen = model.generate(**inp, max_new_tokens=cfg.get("state_tokens", 120), do_sample=False)
        for g in gen:
            txt = proc.decode(g[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append(parse(txt, keys))
    return out


def parse(txt, keys):
    d = {}
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    res = {}
    for k in keys:
        v = d.get(k)
        if v is None:  # fallback: "key: yes"
            mm = re.search(rf"{k}\W+(yes|no|true|false|1|0|\d+)", txt, re.I)
            v = mm.group(1) if mm else 0
        if isinstance(v, str):
            v = {"yes": 1, "true": 1, "no": 0, "false": 0}.get(v.lower(), int(v) if v.isdigit() else 0)
        res[k] = int(v)
    return res


# ---------------- rule engine ----------------

def runs(ts, flags, min_len):
    """Yield (start_t, end_t) of consecutive True runs lasting >= min_len samples."""
    s = None
    for i, f in enumerate(flags + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            if i - s >= min_len:
                yield ts[s], ts[i - 1]
            s = None


def smooth(x, k=1):
    """Majority over a window of 2k+1 samples to suppress single-frame VLM errors."""
    if k <= 0:
        return x
    n = len(x)
    return [sum(x[max(0, i - k):i + k + 1]) * 2 > len(x[max(0, i - k):i + k + 1]) for i in range(n)]


def apply_rules(ts, S, th):
    """ts: timestamps; S: list of per-frame dicts. th: thresholds dict. Returns [(rule_id, t, description)]."""
    if not S:
        return []
    k = th.get("smooth", 1)
    col = lambda key: smooth([bool(s.get(key, 0)) for s in S], k)
    seated, inroom = col("cashier_at_desk"), col("cashier_in_room")
    client, outsider = col("client_present"), col("outsider_cashier_side")
    door, laptop, cash = col("door_open"), col("laptop_screen_on"), col("cash_on_desk")
    safe, personal, idc = col("safe_open"), col("personal_item"), col("id_document")
    counting, bag = col("counting_cash"), col("cash_bag")
    away = [not a for a in seated]
    v = []
    m = th.get("min_frames", 2)

    for a, b in runs(ts, [aw and l for aw, l in zip(away, laptop)], m):
        v.append(("53.26.1", a, "Кассир отошёл от рабочего места, экран компьютера остаётся включённым (не заблокирован)."))
        break
    for a, b in runs(ts, [aw and c for aw, c in zip(away, cash)], m):
        v.append(("231.20.1", a, "Кассир отошёл от рабочего места, наличные остались на столе, а не убраны в сейф."))
        break
    for a, b in runs(ts, [aw and s for aw, s in zip(away, safe)], m):
        v.append(("231.20.1", a, "Сейф открыт в отсутствие кассира."))
        break
    for a, b in runs(ts, [aw and c for aw, c in zip(away, client)], m):
        v.append(("231.22.1", a, "Кассир покинул рабочее место во время обслуживания клиента."))
        break
    for a, b in runs(ts, personal, th.get("personal_frames", 3)):
        v.append(("231.12.1", a, "На столе кассира находятся личные вещи (чашка/сумка/еда)."))
        break
    for a, b in runs(ts, outsider, m):
        v.append(("231.75.1", a, "Посторонний находится на стороне кассира / в кассовой зоне."))
        break
    for a, b in runs(ts, door, th.get("door_frames", 4)):
        v.append(("14.08.1", a, "Дверь кассового помещения остаётся открытой."))
        break
    # ID check: client with cash present, but no ID document seen before cash handling starts
    first_cash = next((i for i, (c, cl) in enumerate(zip(cash, client)) if c and cl), None)
    if first_cash is not None and not any(idc[:first_cash + th.get("id_grace", 1) + 1]):
        v.append(("231.19.1", ts[first_cash], "Кассир принимает наличные без предъявления клиентом документа, удостоверяющего личность."))
    # Bag accepted without counting
    first_bag = next((i for i, b in enumerate(bag) if b), None)
    if first_bag is not None and not any(counting[first_bag:]):
        v.append(("231.24.3", ts[first_bag], "Сумка инкассации принята без пересчёта наличных."))
    return v
