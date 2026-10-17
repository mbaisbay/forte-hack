"""Build a local HTML report of an eval run: per clip label, prediction, timecodes, evidence frames.

The report embeds competition frames, so it stays on team machines (reports/ is git-ignored)
and must never be uploaded to cloud tools. Clips are matched to their newest result by file hash.

Usage (on the GPU box, after eval_dev.py):
  python scripts/report.py --details eval_details_dev_r3.json --out reports/eval_report.html
Open it from the repo's reports/ folder so the video links (../open/videos/...) resolve.
"""

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.media import MediaError, probe  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dev import norm_id  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_results(clips: dict[str, Path], data_dir: Path) -> dict[str, tuple[dict, dict]]:
    """clip id -> (result, meta) of the newest finished job whose upload has the same bytes."""
    by_hash = {sha(p): i for i, p in clips.items()}
    by_size = {p.stat().st_size for p in clips.values()}
    best: dict[str, tuple[float, Path]] = {}
    for up in (data_dir / "uploads").glob("*.mov"):
        if up.stat().st_size not in by_size:
            continue
        clip = by_hash.get(sha(up))
        res = data_dir / "results" / f"{up.stem}.json"
        if clip and res.exists() and (clip not in best or res.stat().st_mtime > best[clip][0]):
            best[clip] = (res.stat().st_mtime, res)
    out = {}
    for clip, (_, res) in best.items():
        meta = res.with_name(f"{res.stem}.meta.json")
        out[clip] = (json.loads(res.read_text()), json.loads(meta.read_text()) if meta.exists() else {})
    return out


def mmss(t: float | None) -> str:
    return "—" if t is None else f"{int(t // 60)}:{t % 60:04.1f}"


def timeline(duration: float, label_t: float | None, meta: dict, events: list[dict]) -> str:
    if duration <= 0:
        return ""
    pct = lambda t: f"{max(0.0, min(100.0, 100 * t / duration)):.2f}%"
    bars = []
    for s, e, conf in meta.get("pass1_intervals", []):
        bars.append(f'<div class="iv p1" style="left:{pct(s)};width:calc({pct(e - s)} + 2px)" title="pass 1 {mmss(s)}–{mmss(e)} conf {conf}"></div>')
    for s, e, status in meta.get("final_intervals", []):
        bars.append(f'<div class="iv st-{status}" style="left:{pct(s)};width:calc({pct(e - s)} + 2px)" title="pass 2 {status} {mmss(s)}–{mmss(e)}"></div>')
    for ev in events:
        bars.append(f'<div class="mark pred" style="left:{pct(ev["timecode_sec"])}" title="prediction {mmss(ev["timecode_sec"])}"></div>')
    if label_t is not None:
        bars.append(f'<div class="mark label" style="left:{pct(label_t)}" title="label {mmss(label_t)}"></div>')
    return f'<div class="tl"><div class="row r1">{"".join(b for b in bars if "p1" in b)}</div><div class="row r2">{"".join(b for b in bars if "p1" not in b)}</div></div>'


def card(clip: str, det: dict, video: Path, res: dict | None, meta: dict, duration: float) -> str:
    lab = det["label"]
    if det.get("verdict") is None:
        outcome = "FAILED"
    else:
        pred = det["verdict"] == "violation"
        outcome = ("TP" if lab["violation"] else "FP") if pred else ("FN" if lab["violation"] else "TN")
    label_txt = f"нарушение в {mmss(lab['time'])}" if lab["violation"] else "без нарушения"
    pred_txt = {"violation": "нарушение", "no_violation": "без нарушения"}.get(det.get("verdict"), f"ошибка: {det.get('status')}")
    vid = f"v-{clip}"
    events = (res or {}).get("violations", [])
    ev_html = []
    for ev in events:
        imgs = "".join(
            f'<figure><img loading="lazy" src="data:image/jpeg;base64,{f["image_base64"]}" alt=""><figcaption>{mmss(f["timestamp_sec"])}</figcaption></figure>'
            for f in ev["evidence_frames"]
        )
        ev_html.append(
            f'<div class="ev"><div class="evh"><button class="tc" data-v="{vid}" data-t="{ev["timecode_sec"]}">{mmss(ev["timecode_sec"])}</button>'
            f'<b>{html.escape(ev["type"])}</b></div><p>{html.escape(ev["description"])}</p><div class="imgs">{imgs}</div></div>'
        )
    if not events:
        ev_html.append('<p class="none">Модель не выдала событий.</p>')
    label_btn = f'<button class="tc lbl" data-v="{vid}" data-t="{lab["time"]}">к метке {mmss(lab["time"])}</button>' if lab["time"] is not None else ""
    src = html.escape("../" + video.as_posix())
    return f"""
<section class="clip" data-outcome="{outcome}">
  <header><h2>{clip}</h2><span class="badge {outcome}">{outcome}</span>
    <span>метка: <b>{label_txt}</b></span><span>прогноз: <b>{pred_txt}</b></span>
    <span class="muted">{det.get("sec", "—")} с · кандидатов p1: {meta.get("raw_candidates", "—")}</span></header>
  <div class="body">
    <div class="left"><video id="{vid}" src="{src}" controls preload="metadata" muted></video>{label_btn}
      {timeline(duration, lab["time"], meta, events)}</div>
    <div class="right">{"".join(ev_html)}</div>
  </div>
</section>"""


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--muted:#6b7280;--line:#e3e6ea;--tp:#15803d;--tn:#2563eb;--fp:#c2410c;--fn:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1a1f25;--fg:#e6e8eb;--muted:#9aa3ad;--line:#2b323a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
.top{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.top h1{font-size:16px;margin:0 12px 0 0}.top button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}
.top button.on{outline:2px solid var(--fg)}
main{padding:16px;display:grid;gap:16px;max-width:1400px;margin:auto}
.clip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.clip header{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:10px}.clip h2{margin:0;font-size:18px}
.badge{font-weight:700;color:#fff;border-radius:5px;padding:1px 8px}.TP{background:var(--tp)}.TN{background:var(--tn)}.FP{background:var(--fp)}.FN{background:var(--fn)}.FAILED{background:#555}
.muted{color:var(--muted)}.body{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}
@media (max-width:900px){.body{grid-template-columns:1fr}}
video{width:100%;border-radius:6px;background:#000}
.tc{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:5px;padding:2px 8px;cursor:pointer;font-variant-numeric:tabular-nums;margin-right:8px}
.lbl{margin-top:6px;border-color:var(--fn)}
.tl{margin-top:8px}.row{position:relative;height:14px;background:var(--bg);border:1px solid var(--line);border-radius:3px;margin-top:3px}
.iv{position:absolute;top:2px;bottom:2px;border-radius:2px}.p1{background:#9ca3af;opacity:.6}
.st-supported{background:var(--tp)}.st-unclear{background:#ca8a04}.st-not_supported{background:#9ca3af}.st-proposed{background:#7c3aed}
.mark{position:absolute;top:-3px;bottom:-3px;width:3px}.mark.label{background:var(--fn)}.mark.pred{background:var(--tn)}
.legend{font-size:12px;color:var(--muted)}
.ev{border-top:1px solid var(--line);padding:8px 0}.ev:first-child{border-top:0;padding-top:0}.evh{display:flex;align-items:center}
.ev p{margin:6px 0}.imgs{display:flex;gap:6px;flex-wrap:wrap}figure{margin:0;width:calc(33% - 4px)}
figure img{width:100%;border-radius:4px;cursor:zoom-in}figcaption{font-size:12px;color:var(--muted)}
.none{color:var(--muted)}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9}#lb img{max-width:96vw;max-height:96vh}
"""

JS = """
document.addEventListener('click',e=>{
  const b=e.target.closest('.tc');
  if(b){const v=document.getElementById(b.dataset.v);v.currentTime=Math.max(0,+b.dataset.t-2);v.play();return}
  const f=e.target.closest('.top button');
  if(f){document.querySelectorAll('.top button').forEach(x=>x.classList.toggle('on',x===f));
    document.querySelectorAll('.clip').forEach(c=>c.style.display=(f.dataset.f==='all'||c.dataset.outcome===f.dataset.f)?'':'none');return}
  if(e.target.matches('figure img')){const lb=document.getElementById('lb');lb.querySelector('img').src=e.target.src;lb.style.display='flex';return}
  if(e.target.closest('#lb'))e.target.closest('#lb').style.display='none';
});
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", type=Path, required=True)
    ap.add_argument("--videos", type=Path, default=Path("open/videos"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("reports/eval_report.html"))
    args = ap.parse_args()

    details = json.loads(args.details.read_text(encoding="utf-8"))
    clips = {norm_id(p.name): p for p in sorted(args.videos.glob("*.mov")) if norm_id(p.name) in details}
    results = latest_results(clips, args.data_dir)

    order = {"FN": 0, "FP": 1, "FAILED": 2, "TP": 3, "TN": 4}
    cards, counts = [], {}
    for clip in sorted(clips, key=lambda c: int(c.split("-")[1])):
        res, meta = results.get(clip, (None, {}))
        try:
            duration = probe(clips[clip]).duration_sec
        except MediaError:
            duration = 0.0
        html_card = card(clip, details[clip], clips[clip], res, meta, duration)
        outcome = html_card.split('data-outcome="', 1)[1].split('"', 1)[0]
        counts[outcome] = counts.get(outcome, 0) + 1
        cards.append((order[outcome], html_card))
    cards.sort(key=lambda x: x[0])

    buttons = '<button class="on" data-f="all">все</button>' + "".join(
        f'<button data-f="{k}">{k} ({counts[k]})</button>' for k in order if k in counts)
    legend = ('<span class="legend">шкала: верхняя полоса — кандидаты 1-го прохода; нижняя — 2-й проход '
              '(зелёный supported, жёлтый unclear, серый not_supported); красная черта — метка, синяя — прогноз</span>')
    page = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Eval report</title><style>{CSS}</style></head><body>'
            f'<div class="top"><h1>{html.escape(args.details.name)}</h1>{buttons}{legend}</div>'
            f'<main>{"".join(c for _, c in cards)}</main><div id="lb"><img alt=""></div><script>{JS}</script></body></html>')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"clips={len(cards)} matched_results={len(results)} outcomes={dict(sorted(counts.items()))} "
          f"size={args.out.stat().st_size // 1024} KB -> {args.out}")


if __name__ == "__main__":
    main()
