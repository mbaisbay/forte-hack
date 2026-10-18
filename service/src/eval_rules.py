"""How well does the LLM extractor rebuild our catalogue? Runs it on the rules catalogue PDF and compares
with data/rules.json + the tree mapping (our hand-made reference).
  python src/eval_rules.py path/to/Каталог_правил.pdf
"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import rules_llm as L

ROOT = Path(__file__).resolve().parent.parent
# reference: (document, paragraph) -> tree leaves that detect it (empty = in catalogue, no detector)
REF = {("231", "12"): {"personal_items"}, ("231", "19"): {"no_id"}, ("231", "20"): {"next_client", "bag_left", "safe_open"},
       ("231", "21"): set(), ("231", "22"): set(), ("231", "24"): set(), ("231", "43"): set(), ("231", "46"): set(),
       ("231", "75"): set(), ("14", "8"): {"door_open"}, ("14", "16"): {"door_open"}, ("14", "21"): set()}


def key(r):
    d = re.search(r"№\s*(\d+)", r["source"]); p = re.match(r"\d+", r["paragraph"] or "")
    return (d.group(1) if d else "?", p.group(0) if p else "?")


if __name__ == "__main__":
    tree = json.loads((ROOT / "data/tree.json").read_text(encoding="utf-8"))
    text = L.read_text(sys.argv[1]); t0 = time.time()
    # every "Пункт N" header in the catalogue, with the document it falls under
    allp, doc = set(), "?"
    for line in text.splitlines():
        m = re.match(r"\s*Правила\s*№\s*(\d+)", line); doc = m.group(1) if m else doc  # headers only, not in-text references
        for p in re.findall(r"Пункт\s*(\d+)", line):
            allp.add((doc, p))
    cache = ROOT / "data/eval_rules.json"
    if "--cached" in sys.argv:  # re-score the last extraction without the GPU
        got, sec = json.loads(cache.read_text(encoding="utf-8")), json.loads((ROOT / "data/eval_rules_report.json").read_text(encoding="utf-8"))["seconds"]
    else:
        got = []
        for u in L.units(text):
            got += L.extract(u, Path(sys.argv[1]).name, tree)
        sec = time.time() - t0
        cache.write_text(json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
    keys = {key(r) for r in got}
    vis = {key(r) for r in got if r["visual"]}
    cov = len(allp & keys) / len(allp)
    rec = len(set(REF) & vis) / len(REF)
    prec = len(set(REF) & vis) / max(1, len(vis))
    mapped = [k for k in REF if REF[k]]
    node_ok = sum(any(key(r) == k and r["node"] in REF[k] for r in got) for k in mapped)
    rep = {"model": Path(L.MODEL).name, "ref_paragraphs": len(REF), "seconds": round(sec), "paragraphs_in_doc": len(allp), "extracted": len(got),
                      "coverage": round(cov, 2), "visual_recall_vs_ours": round(rec, 2),
                      "visual_precision_vs_ours": round(prec, 2), "node_mapping": f"{node_ok}/{len(mapped)}",
                      "missed": sorted(allp - keys), "visual_not_in_ours": sorted(vis - set(REF))}
    (ROOT / "data/eval_rules_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
