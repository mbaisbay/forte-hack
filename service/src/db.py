"""SQLite storage: analysed videos (+ human labels) and uploaded regulatory documents (+ LLM rule suggestions).
Rules catalogue and decision tree are JSON files in data/ (small, human-edited, easy to diff)."""
import json, sqlite3, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "storage"
(STORE / "videos").mkdir(parents=True, exist_ok=True)
(STORE / "docs").mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()
_con = sqlite3.connect(STORE / "service.db", check_same_thread=False)
_con.row_factory = sqlite3.Row
_con.executescript("""
CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY, filename TEXT, path TEXT, created REAL, status TEXT,
  result TEXT, error TEXT, label TEXT);
CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, filename TEXT, path TEXT, created REAL, status TEXT,
  chunks_total INTEGER DEFAULT 0, chunks_done INTEGER DEFAULT 0, suggestions TEXT DEFAULT '[]', error TEXT);
""")
JSON_COLS = {"result", "label", "suggestions"}


def _row(r):
    return None if r is None else {k: (json.loads(r[k]) if k in JSON_COLS and r[k] else r[k]) for k in r.keys()}


def insert(table, **kw):
    kw.setdefault("created", time.time())
    with _lock:
        _con.execute(f"INSERT INTO {table} ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})", list(kw.values()))
        _con.commit()


def update(table, id, **kw):
    kw = {k: json.dumps(v, ensure_ascii=False) if k in JSON_COLS and v is not None else v for k, v in kw.items()}
    with _lock:
        _con.execute(f"UPDATE {table} SET {','.join(k + '=?' for k in kw)} WHERE id=?", [*kw.values(), id])
        _con.commit()


def get(table, id):
    with _lock:
        return _row(_con.execute(f"SELECT * FROM {table} WHERE id=?", (id,)).fetchone())


def all(table, where=""):
    with _lock:
        return [_row(r) for r in _con.execute(f"SELECT * FROM {table} {where} ORDER BY created DESC").fetchall()]


def add_suggestions(id, items):
    """Append one chunk's suggestions atomically (chunks are processed one by one by the worker)."""
    with _lock:
        r = _con.execute("SELECT suggestions, chunks_done, chunks_total FROM docs WHERE id=?", (id,)).fetchone()
        s = json.loads(r["suggestions"]) + items
        done = r["chunks_done"] + 1
        _con.execute("UPDATE docs SET suggestions=?, chunks_done=?, status=? WHERE id=?",
                     (json.dumps(s, ensure_ascii=False), done, "done" if done >= r["chunks_total"] else "processing", id))
        _con.commit()


def read_json(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def write_json(name, obj):
    p = ROOT / "data" / name
    if p.exists():  # keep one backup so a bad edit can be undone by hand
        p.with_suffix(".bak.json").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
