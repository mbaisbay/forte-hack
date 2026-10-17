"""Local HTML viewer of a logged run: per clip, every model call in time order with the frames the model saw,
its raw answer, and the final events. One page per clip plus an index. Stays on this machine.

Usage:
  python scripts/calls_report.py --details eval_details_dev_r14_both.json --calls-dir calls_r14 --out reports/calls_r14
  then open http://127.0.0.1:8765/reports/calls_r14/index.html  (scripts/serve_report.py)
"""

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import Settings  # noqa: E402
from app.media import decode  # noqa: E402
from eval_dev import load_labels, norm_id  # noqa: E402

KIND_RU = {"observe": "Скан окна", "verify": "Перепроверка", "checklist": "Чек-лист", "recheck": "Узкий вопрос", "states": "Состояния"}
CSS = """
:root{--bg:#f4f6f8;--card:#fff;--fg:#17212b;--muted:#5b6875;--line:#d5dbe2;--acc:#1f5fa8;--bad:#b3261e;--good:#1e7a46}
@media (prefers-color-scheme:dark){:root{--bg:#10161c;--card:#18212a;--fg:#e5eaf0;--muted:#98a5b3;--line:#2b3743;--acc:#79aef0;--bad:#f2867c;--good:#6fd29a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:16px}
a{color:var(--acc)}h1{font-size:20px;margin:0 0 8px}h2{font-size:16px;margin:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0;display:grid;gap:8px}
.head{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.pill{font:12px ui-monospace,monospace;padding:1px 8px;border-radius:99px;background:var(--bg);border:1px solid var(--line)}
.bad{color:var(--bad)}.good{color:var(--good)}.muted{color:var(--muted)}
.strip{display:flex;gap:4px;overflow-x:auto;padding-bottom:4px}.strip figure{margin:0;flex:0 0 auto;width:96px}
.strip img{width:96px;border-radius:3px;cursor:zoom-in;display:block}.strip figcaption{font:10px ui-monospace,monospace;color:var(--muted)}
pre{margin:0;max-height:260px;overflow:auto;font:12px/1.4 ui-monospace,monospace;background:var(--bg);padding:8px;border-radius:6px;white-space:pre-wrap}
table{border-collapse:collapse}td,th{padding:3px 10px;border-bottom:1px solid var(--line);text-align:left}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center}#lb img{max-height:96vh;max-width:96vw}
"""
JS = """document.addEventListener('click',e=>{const lb=document.getElementById('lb');
if(e.target.matches('.strip img')){lb.querySelector('img').src=e.target.src;lb.style.display='flex'}else if(e.target.closest('#lb'))lb.style.display='none'});"""


def thumb(jpeg: bytes, width: int = 192) -> str:
    im = Image.open(io.BytesIO(jpeg))
    im.thumbnail((width, width * 2))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def page(clip: str, det: dict, calls: list[dict], video, team: dict, org: dict | None) -> str:
    by_id = {f.frame_id: f for f in video.frames}
    cache: dict[str, str] = {}

    def img(fid: str) -> str:
        if fid not in cache and fid in by_id:
            cache[fid] = thumb(by_id[fid].model_jpeg)
        return cache.get(fid, "")

    lab = "нарушение " + ", ".join(f"{e['t']:.0f} с ({', '.join(e['codes'])})" for e in team["events"]) if team["violation"] else "без нарушения"
    org_txt = ("нарушение " + ", ".join(f"{t:.0f} с" for t in org["times"])) if org and org["violation"] else "без нарушения"
    pred = det.get("verdict") == "violation"
    ok = pred == team["violation"]
    out = [f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{clip} — вызовы модели</title><style>{CSS}</style></head><body>",
           f"<p><a href='index.html'>← все ролики</a></p><h1>{clip} · {video.info.duration_sec:.1f} с</h1>",
           f"<div class='card'><div class='head'><span>Разметка команды: <b>{html.escape(lab)}</b></span>"
           f"<span>Организаторы: <b>{html.escape(org_txt)}</b></span>"
           f"<span class='pill {'good' if ok else 'bad'}'>сервис: {det.get('verdict')} — {'верно' if ok else 'ошибка'}</span></div>",
           "<table><tr><th>Таймкод</th><th>Итоговое событие</th></tr>"]
    for e in det.get("events", []):
        out.append(f"<tr><td>{e['t']:.1f} с</td><td>{html.escape(e['type'])}</td></tr>")
    if not det.get("events"):
        out.append("<tr><td colspan=2 class='muted'>событий нет</td></tr>")
    out.append("</table></div>")
    for i, c in enumerate(sorted(calls, key=lambda c: (c["t"] or [0])[0])):
        t = c.get("t") or [0, 0]
        try:
            pretty = json.dumps(json.loads(c["content"]), ensure_ascii=False, indent=1) if c.get("content") else "(пусто)"
        except json.JSONDecodeError:
            pretty = c.get("content") or ""
        frames = c.get("frames", [])
        strip = "".join(f"<figure><img loading='lazy' src='{img(f)}' alt='{f}'><figcaption>{f} · {by_id[f].timestamp_sec:.1f}с</figcaption></figure>"
                        for f in frames if f in by_id)
        out.append(f"<div class='card'><div class='head'><h2>#{i + 1} {KIND_RU.get(c['kind'], c['kind'])}</h2>"
                   f"<span class='pill'>{t[0]:.1f}–{t[1]:.1f} с</span><span class='pill'>{len(frames)} кадров</span>"
                   f"<span class='pill'>{c.get('sec', 0):.1f} с ответ</span><span class='pill'>{c.get('finish')}</span></div>"
                   f"<div class='strip'>{strip}</div>"
                   + (f"<details open><summary>Рассуждение модели ({len(c['reasoning'])} симв.)</summary>"
                      f"<pre>{html.escape(c['reasoning'])}</pre></details>" if c.get("reasoning") else "")
                   + f"<pre>{html.escape(pretty)}</pre></div>")
    out.append(f"<div id='lb'><img alt=''></div><script>{JS}</script></body></html>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", type=Path, required=True)
    ap.add_argument("--calls-dir", type=Path, required=True, help="folder with <job_id>.calls.jsonl")
    ap.add_argument("--videos", type=Path, default=Path("open/videos"))
    ap.add_argument("--labels", type=Path, default=Path("labels_fixed.xlsx"))
    ap.add_argument("--org-labels", type=Path, default=Path("open/open_labels.xlsx"))
    ap.add_argument("--out", type=Path, default=Path("reports/calls"))
    args = ap.parse_args()
    s = Settings()
    details = json.loads(args.details.read_text(encoding="utf-8"))
    team, org = load_labels(args.labels), load_labels(args.org_labels)
    clips = {norm_id(p.name): p for p in args.videos.glob("*.mov")}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in sorted(details, key=lambda c: int(c.split("-")[1])):
        det = details[clip]
        path = args.calls_dir / f"{det.get('job_id')}.calls.jsonl"
        if clip not in clips or not path.exists():
            continue
        calls = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        video = decode(clips[clip], s.verify_fps, s.model_max_side)
        (args.out / f"{clip}.html").write_text(page(clip, det, calls, video, team[clip], org.get(clip)), encoding="utf-8")
        pred = det.get("verdict") == "violation"
        kind = ("верно" if pred == team[clip]["violation"] else "ложное срабатывание" if pred else "пропуск")
        rows.append(f"<tr><td><a href='{clip}.html'>{clip}</a></td><td>{'нарушение' if team[clip]['violation'] else 'чистый'}</td>"
                    f"<td>{det.get('verdict')}</td><td class='{'good' if kind == 'верно' else 'bad'}'>{kind}</td>"
                    f"<td>{len(det.get('events', []))}</td><td>{len(calls)}</td></tr>")
        print("page", clip, flush=True)
    (args.out / "index.html").write_text(
        f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>Вызовы модели</title><style>{CSS}</style></head><body>"
        f"<h1>Вызовы модели по роликам ({html.escape(args.details.name)})</h1><div class='card'><table><tr><th>Ролик</th><th>Разметка</th>"
        f"<th>Вердикт сервиса</th><th>Итог</th><th>Событий</th><th>Вызовов</th></tr>{''.join(rows)}</table></div></body></html>",
        encoding="utf-8")


if __name__ == "__main__":
    main()
