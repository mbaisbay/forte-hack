"""Regulatory document -> suggested rules with a local text LLM (Qwen3-8B, 4-bit, ~6 GB VRAM).
Code splits the document into numbered paragraphs and builds the source reference itself (document, chapter, paragraph),
so references cannot be invented; the LLM only judges each paragraph: requirement? visible on video? which check?
  python src/rules_llm.py doc.pdf            # print suggestions
"""
import json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("RULES_LLM", str(Path(os.environ.get("MODELS_DIR", ROOT / "models")) / "Qwen3-8B"))
_m = {}

SYS = ("You analyse Kazakhstan banking regulations for a CCTV compliance system that watches one bank cash desk "
       "from a ceiling camera (no audio; it cannot read documents, amounts or screens). Answer with JSON only.")
TASK = """DOCUMENT: {doc}
SECTION: {source}

EXISTING CAMERA CHECKS (node id: what it detects):
{nodes}

PARAGRAPH:
<<<
{text}
>>>

Decide whether this paragraph states a requirement for the cashier, the cash room or cash handling, and whether its violation
would be VISIBLE on the video. Visible: people and actions (client present, cashier leaves the desk, counting in front of the client,
showing an ID document, outsiders in the room), objects (cash or personal items left on the desk, cash bag, safe or door open, laptop left on).
Not visible: signatures, amounts, document contents, computer entries, staffing rules, packaging details.
Return JSON only:
{{"requirement": true or false,
  "title": "short Russian name of the violation (max 8 words)",
  "quote": "the key sentence copied verbatim from PARAGRAPH",
  "visual": true or false,
  "check": "Russian, one sentence: what the camera would see when it is violated (empty if not visual)",
  "node": "id of the existing check that detects it, or none"}}"""

HEAD = re.compile(r"^\s*(?:Правила|Постановление|Приказ|Закон|Требования)[^\n]*?№\s*\d+[^\n]*", re.I)
PARA = re.compile(r"(?:^|·)\s*(?:Пункт\s*(\d+)(?:,\s*подпункт\s*(\d+)\))?\s*$|^\s*(\d{1,3})\.\s+\S)")


def read_text(path):
    p = Path(path); ext = p.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        t = "\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
    elif ext == ".docx":
        import docx
        t = "\n".join(par.text for par in docx.Document(str(p)).paragraphs)
    else:
        t = p.read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r" ([,.;:)])", r"\1", t).replace("( ", "(")


def units(text, size=2500):
    """Numbered paragraphs with code-built sources; plain chunks if the text has no numbering."""
    out, src, cur, num = [], "", [], None

    def flush():
        if num and cur:
            out.append({"source": f"{src}, п.{num}" if src else f"п.{num}", "paragraph": num, "text": "".join(cur)[:size]})

    headers = "Пункт" in text  # catalogue style: only "Пункт N" starts a paragraph; adilet style: every "N." does
    for line in text.splitlines(keepends=True):
        h = HEAD.match(line)
        if h:
            flush(); cur, num = [], None
            src = re.split(r"\.\s", h.group(0))[0].replace(" ·", ",").strip()
            continue
        m = PARA.search(line)
        if m and (m.group(1) if headers else m.group(3)):
            flush(); cur = []
            num = m.group(1) + (f".{m.group(2)}" if m.group(2) else "") if headers else m.group(3)
        if num:
            cur.append(line)
    flush()
    if len(out) >= 2:
        return out
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > size and buf:
            chunks.append(buf); buf = ""
        buf += line
    if buf.strip():
        chunks.append(buf)
    return [{"source": "", "paragraph": str(i + 1), "text": c} for i, c in enumerate(chunks)]


def leaf_nodes(tree):
    parents = {n["parent"] for n in tree["nodes"]}
    return [n for n in tree["nodes"] if n["id"] not in parents and n["id"] != "client_fallback"]


def load():
    if not _m:
        import torch, vlm
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        vlm.free()
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        _m["tok"] = AutoTokenizer.from_pretrained(MODEL)
        _m["model"] = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                           quantization_config=q).eval()
    return _m["model"], _m["tok"]


def free():
    if _m:
        import torch
        _m.clear(); torch.cuda.empty_cache()


def extract(unit, doc, tree):
    """One unit from units() -> list with 0 or 1 suggestion."""
    import torch
    model, tok = load()
    nodes = "\n".join(f"- {n['id']}: {n['title']}. {n['observe']}" for n in leaf_nodes(tree))
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": TASK.format(doc=doc, source=unit["source"] or "?", nodes=nodes, text=unit["text"])}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                  enable_thinking=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=400, do_sample=False)
    txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        r = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        r = {}
    if not r.get("requirement") or not r.get("quote"):
        return []
    ids = {n["id"] for n in leaf_nodes(tree)}
    return [{"source": unit["source"] or doc, "paragraph": unit["paragraph"], "title": str(r.get("title", "")),
             "quote": str(r["quote"]), "visual": bool(r.get("visual")), "check": str(r.get("check", "")),
             "node": r.get("node") if r.get("node") in ids else "none"}]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    tree = json.loads((ROOT / "data/tree.json").read_text(encoding="utf-8"))
    for u in units(read_text(sys.argv[1])):
        for r in extract(u, Path(sys.argv[1]).name, tree):
            print(json.dumps(r, ensure_ascii=False))
