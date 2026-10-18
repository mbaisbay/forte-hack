"""Text-only judge: reads a narration timeline + rule catalogue -> violations JSON.
Output: data/judge_<narr>_<judge>/<video>.json
  python src/judge.py --narr gemma4b --model models/Qwen3-8B --quant 4bit --tag q8
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
SYS = ("You are a strict bank cash-desk compliance auditor. You get a text timeline written by a camera observer "
       "(it can contain small mistakes) and a catalogue of rules. Decide whether any rule is violated. "
       "Recall matters most: if the timeline shows a plausible violation, report it. "
       "Every violation must quote the timeline line it is based on.")
TASK = """RULES:
{RULES}

TIMELINE:
{TL}

Typical correct procedure: client enters, shows ID card, hands cash, cashier counts (machine or by hand), checks the laptop, puts cash in the safe and closes it, hands a receipt, client leaves. When the cashier leaves the room: laptop locked/off, safe closed, no cash or personal items on the desk, door closed.

Return JSON only:
{{"violation": true|false, "violations": [{{"rule_id": "<id>", "t": <seconds>, "evidence": "<quoted timeline line>", "reason": "<short, Russian>"}}]}}"""


def rules_text():
    R = json.loads((ROOT / "data/rules.json").read_text(encoding="utf-8"))
    return "\n".join(f"[{r['id']}] {r['title']}: {r['check']}" for r in R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--narr", default="gemma4b")
    ap.add_argument("--model", default="models/Qwen3-8B")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--tag", default="q8")
    ap.add_argument("--think", action="store_true")
    a = ap.parse_args()
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if a.quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    src = ROOT / f"data/narr_{a.narr}"
    d = ROOT / f"data/judge_{a.narr}_{a.tag}"; d.mkdir(parents=True, exist_ok=True)
    R = rules_text()
    for f in sorted(src.glob("*.txt")):
        v = f.stem
        if a.videos and v not in a.videos or (d / f"{v}.json").exists():
            continue
        t0 = time.time()
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": TASK.format(RULES=R, TL=f.read_text(encoding="utf-8"))}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                      enable_thinking=a.think).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=2048 if a.think else 512, do_sample=False)
        txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        body = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
        m = re.search(r"\{.*\}", body, re.S)
        try:
            js = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            js = {"violation": "true" in body.lower()[:200], "parse_error": True}
        js["raw"] = txt
        (d / f"{v}.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
        print(v, js.get("violation"), [x.get("rule_id") for x in js.get("violations", []) if isinstance(x, dict)],
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
