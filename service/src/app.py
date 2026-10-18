"""Service: 3 pages (videos, rules, decision tree) + jury API (case section 10).
  API_KEY=... python -m uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
Jury API (X-API-Key):  POST /api/v1/analyze (file=.mov) -> {"job_id"};  GET /api/v1/jobs/{job_id}
UI API (no key, local): /api/ui/...
"""
import json, os, re, uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

import db, worker

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).parent / "static"
API_KEY = os.environ.get("API_KEY", "change-me")
app = FastAPI(title="ForteBank cash-desk violation detector")
worker.start()


def check(key):
    if key != API_KEY:
        raise HTTPException(401, "invalid API key")


def found(x, what="not found"):
    if x is None:
        raise HTTPException(404, what)
    return x


async def save_video(file):
    vid = uuid.uuid4().hex[:8]
    path = db.STORE / "videos" / f"{vid}{Path(file.filename or 'v.mov').suffix.lower() or '.mov'}"
    path.write_bytes(await file.read())
    db.insert("videos", id=vid, filename=file.filename or path.name, path=str(path), status="processing")
    worker.put_video(vid)
    return vid


def jury_view(v):
    out = {"job_id": v["id"], "status": v["status"]}
    if v["status"] == "done":
        r = v["result"]
        out |= {"verdict": r["verdict"], "violations": r["violations"]}
    return out


# ---------- jury API ----------
@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check(x_api_key)
    return {"job_id": await save_video(file)}


@app.get("/api/v1/jobs/{jid}")
def job(jid: str, x_api_key: str = Header(None)):
    check(x_api_key)
    return jury_view(found(db.get("videos", jid), "unknown job"))


# ---------- videos + labels ----------
@app.post("/api/ui/videos")
async def ui_upload(file: UploadFile = File(...)):
    return {"id": await save_video(file)}


@app.get("/api/ui/videos")
def ui_videos():
    out = []
    for v in db.all("videos"):
        r = v["result"] or {}
        out.append({"id": v["id"], "filename": v["filename"], "created": v["created"], "status": v["status"],
                    "verdict": r.get("verdict"), "n": len(r.get("violations", [])), "label": (v["label"] or {}).get("verdict")})
    return out


@app.get("/api/ui/videos/{vid}")
def ui_video(vid: str):
    v = found(db.get("videos", vid)); v.pop("path")
    return v


@app.get("/api/ui/videos/{vid}/file")
def ui_video_file(vid: str):
    return FileResponse(found(db.get("videos", vid))["path"])


@app.put("/api/ui/videos/{vid}/label")
def ui_label(vid: str, label: dict = Body(...)):
    found(db.get("videos", vid)); db.update("videos", vid, label=label)
    return {"ok": True}


@app.post("/api/ui/videos/{vid}/rerun")
def ui_rerun(vid: str):
    found(db.get("videos", vid)); db.update("videos", vid, status="processing", error=None); worker.put_video(vid)
    return {"ok": True}


@app.get("/api/ui/labels")
def ui_labels():
    """All human labels next to model output (for measuring quality / re-tuning thresholds)."""
    return [{"id": v["id"], "filename": v["filename"], "label": v["label"],
             "predicted": [{k: x[k] for k in ("rule_id", "check", "timecode_sec")} for x in (v["result"] or {}).get("violations", [])]}
            for v in db.all("videos", "WHERE label IS NOT NULL")]


# ---------- rules catalogue + documents ----------
@app.get("/api/ui/rules")
def ui_rules():
    return db.read_json("rules.json")


@app.put("/api/ui/rules")
def ui_rules_put(rules: list = Body(...)):
    ids = [r.get("id") for r in rules]
    if not all(ids) or len(set(ids)) != len(ids):
        raise HTTPException(400, "every rule needs a unique id")
    db.write_json("rules.json", rules)
    return {"ok": True}


@app.post("/api/ui/docs")
async def ui_doc(file: UploadFile = File(...)):
    import rules_llm
    did = uuid.uuid4().hex[:8]
    path = db.STORE / "docs" / f"{did}{Path(file.filename or 'd.txt').suffix.lower()}"
    path.write_bytes(await file.read())
    try:
        ch = rules_llm.units(rules_llm.read_text(path))
    except Exception as e:
        raise HTTPException(400, f"cannot read document: {e}")
    (db.STORE / "docs" / f"{did}.chunks.json").write_text(json.dumps(ch, ensure_ascii=False), encoding="utf-8")
    db.insert("docs", id=did, filename=file.filename or path.name, path=str(path),
              status="processing" if ch else "done", chunks_total=len(ch))
    worker.put_doc(did, len(ch))
    return {"id": did}


@app.get("/api/ui/docs")
def ui_docs():
    """Suggestions for a paragraph already in the catalogue are marked with the existing rule id ("exists")."""
    have = {}
    for r in db.read_json("rules.json"):
        d = re.search(r"№\s*(\d+)", r["source"])
        for p in re.findall(r"п\.\s*(\d+)", r["source"]):
            have.setdefault((d.group(1) if d else "?", p), []).append(r["id"])
    out = []
    for d in db.all("docs"):
        for s in d["suggestions"]:
            m = re.search(r"№\s*(\d+)", s["source"]); p = re.match(r"\d+", s["paragraph"])
            s["exists"] = have.get((m.group(1) if m else "?", p.group(0) if p else "?"), [])
        out.append({k: v for k, v in d.items() if k != "path"})
    return out


def new_rule_id(s, rules):
    d = re.search(r"№\s*(\d+)", s.get("source", "")); p = re.match(r"\d+", s.get("paragraph", ""))
    base = f"{d.group(1) if d else 'doc'}.{p.group(0) if p else 'x'}"
    ids, k = {r["id"] for r in rules}, 1
    while f"{base}.{k}" in ids:
        k += 1
    return f"{base}.{k}"


@app.post("/api/ui/docs/{did}/suggestions/{i}")
def ui_suggestion(did: str, i: int, body: dict = Body(...)):
    """body: {"action": "accept"|"reject", "edit": {title, source, quote, check}} -> accept adds the rule to the catalogue."""
    d = found(db.get("docs", did)); s = d["suggestions"]
    if not 0 <= i < len(s):
        raise HTTPException(404, "no such suggestion")
    s[i] |= body.get("edit", {})
    if body.get("action") == "accept":
        rules = db.read_json("rules.json")
        rid = body.get("rule_id") or new_rule_id(s[i], rules)
        if rid in {r["id"] for r in rules}:
            raise HTTPException(400, f"rule {rid} already exists")
        rules.append({"id": rid, "source": s[i]["source"], "title": s[i]["title"], "text": s[i]["quote"],
                      "check": s[i]["check"], "fewshot": [], "origin": d["filename"]})
        db.write_json("rules.json", rules)
        s[i] |= {"status": "accepted", "rule_id": rid}
    else:
        s[i]["status"] = "rejected"
    db.update("docs", did, suggestions=s)
    return s[i]


@app.get("/api/ui/eval_rules")
def ui_eval_rules():
    p = ROOT / "data/eval_rules_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------- decision tree ----------
@app.get("/api/ui/tree")
def ui_tree():
    return db.read_json("tree.json")


@app.put("/api/ui/tree")
def ui_tree_put(tree: dict = Body(...)):
    ids = {n["id"] for n in tree.get("nodes", [])}
    if not ids or any(n["parent"] not in ids | {None} for n in tree["nodes"]):
        raise HTTPException(400, "bad tree")
    known = {r["id"] for r in db.read_json("rules.json")}
    bad = sorted({r for n in tree["nodes"] for r in n["rule_ids"]} - known)
    if bad:
        raise HTTPException(400, f"unknown rule ids: {bad}")
    db.write_json("tree.json", tree)
    return {"ok": True}


@app.get("/api/ui/mapping_suggestions")
def ui_mapping():
    """LLM-proposed rule -> check links (accepted rules only) that are not yet in the tree."""
    tree = db.read_json("tree.json"); have = {(r, n["id"]) for n in tree["nodes"] for r in n["rule_ids"]}
    out = []
    for d in db.all("docs"):
        for s in d["suggestions"]:
            if s.get("status") == "accepted" and s.get("node", "none") != "none" and (s["rule_id"], s["node"]) not in have:
                out.append({"rule_id": s["rule_id"], "node": s["node"], "title": s["title"], "check": s["check"]})
    return out


# ---------- pages ----------
@app.get("/")
def page_videos():
    return FileResponse(STATIC / "videos.html")


@app.get("/rules")
def page_rules():
    return FileResponse(STATIC / "rules.html")


@app.get("/tree")
def page_tree():
    return FileResponse(STATIC / "tree.html")


@app.get("/static/{name}")
def static(name: str):
    p = STATIC / Path(name).name
    return FileResponse(found(p if p.exists() else None))
