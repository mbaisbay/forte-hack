"""One GPU worker thread. Videos have priority over document chunks, so a long document never
blocks a jury video for more than one chunk (~1 min). Unfinished jobs are re-queued on restart."""
import itertools, json, queue, threading, traceback

import db

Q: "queue.PriorityQueue" = queue.PriorityQueue()
_seq = itertools.count()
VIDEO, DOC = 0, 1


def put_video(vid):
    Q.put((VIDEO, next(_seq), vid, None))


def put_doc(did, n_chunks, start=0):
    for i in range(start, n_chunks):
        Q.put((DOC, next(_seq), did, i))


def _video(vid):
    import detect, rules_llm
    rules_llm.free()
    v = db.get("videos", vid)
    try:
        res = detect.analyze(v["path"])
        db.update("videos", vid, status="done", result=res, error=None)
    except Exception:
        db.update("videos", vid, status="failed", error=traceback.format_exc()[-800:])


def _doc(did, i):
    import rules_llm
    d = db.get("docs", did)
    try:
        chunk = json.loads((db.STORE / "docs" / f"{did}.chunks.json").read_text(encoding="utf-8"))[i]
        items = rules_llm.extract(chunk, d["filename"], db.read_json("tree.json"))
        db.add_suggestions(did, [x | {"status": "pending", "chunk": i} for x in items])
    except Exception:  # skip the broken chunk, keep the rest of the document
        db.add_suggestions(did, [])
        db.update("docs", did, error=traceback.format_exc()[-800:])


def loop():
    while True:
        kind, _, id, i = Q.get()
        _video(id) if kind == VIDEO else _doc(id, i)


def start():
    for v in db.all("videos", "WHERE status='processing'"):
        put_video(v["id"])
    for d in db.all("docs", "WHERE status='processing'"):
        put_doc(d["id"], d["chunks_total"], d["chunks_done"])
    threading.Thread(target=loop, daemon=True).start()
