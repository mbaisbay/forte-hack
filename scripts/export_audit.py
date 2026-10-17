"""Export what Qwen saw and answered for each DEV clip, for an offline audit of prompt/model/approach.

Per clip writes audit/<clip>/: the checklist-call frames (exact model JPEGs), Qwen's raw checklist answer
per segment, the findings code derived from it, the service's final events (from an eval details file)
and the label with the team's note. Holdout clips are skipped so they stay an unbiased check.
Only open/videos is read: jury uploads must never be exported.

Usage (on the GPU box, VLM server running):
  python scripts/export_audit.py --details eval_details_dev_r5_both.json --out audit
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import checklist  # noqa: E402
from app.config import Settings  # noqa: E402
from app.media import decode  # noqa: E402
from app.vlm import build_vlm, text_part  # noqa: E402
from app.pipeline import frames_parts  # noqa: E402
from eval_dev import HOLDOUT, load_labels, norm_id  # noqa: E402


def label_notes(path: Path) -> dict[str, list[dict]]:
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    notes: dict[str, list[dict]] = {}
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if row[0] and len(row) > 4 and (row[3] or row[4]):
            notes.setdefault(norm_id(row[0]), []).append({"time": row[1], "codes": row[3], "note": row[4]})
    return notes


async def export_clip(clip: str, path: Path, out: Path, s: Settings, vlm, prompt: str, extra: dict) -> None:
    video = decode(path, s.verify_fps, s.model_max_side)
    d = out / clip
    d.mkdir(parents=True, exist_ok=True)
    segs = checklist.segments(video, s, 0)
    raw, found = [], []
    for k, frames in enumerate(segs):
        for f in frames:
            (d / f"seg{k}_{f.frame_id}_t{f.timestamp_sec:06.2f}.jpg").write_bytes(f.model_jpeg)
        ans = await vlm.chat_json([text_part(prompt)] + frames_parts(frames), checklist.checklist_schema(), max_tokens=1500)
        raw.append({"segment": k, "frames": [f"{f.frame_id} t={f.timestamp_sec:.2f}s" for f in frames], "answer": ans})
        found += [{"rule_id": x.rule_id, "t": round(x.t, 2), "frame_id": x.frame_id} for x in checklist.findings_from(ans, frames, video)]
    (d / "case.json").write_text(json.dumps({
        "clip": clip, "duration_sec": round(video.info.duration_sec, 2),
        **extra, "qwen_checklist_raw": raw, "checklist_findings_from_raw": found,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


async def main_async(args) -> None:
    s = Settings()
    vlm = build_vlm(s)
    prompt = (s.prompts_dir / "checklist.txt").read_text(encoding="utf-8")
    labels = load_labels(args.labels)
    notes = label_notes(args.labels)
    details = json.loads(args.details.read_text(encoding="utf-8"))
    clips = {norm_id(p.name): p for p in sorted(args.videos.glob("*.mov"))}
    todo = sorted((c for c in details if c in clips and c not in HOLDOUT), key=lambda c: int(c.split("-")[1]))
    if args.only:
        todo = [c for c in todo if c in set(args.only.split(","))]
    for clip in todo:
        det = details[clip]
        lab = labels[clip]
        extra = {
            "label": {"violation": lab["violation"], "times_sec": lab["times"], "codes": lab["codes"], "rows": notes.get(clip, [])},
            "service_result": {"verdict": det.get("verdict"), "events": det.get("events", [])},
        }
        await export_clip(clip, clips[clip], args.out, s, vlm, prompt, extra)
        print(f"exported {clip}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("open/labels_fixed.xlsx"))
    ap.add_argument("--videos", type=Path, default=Path("open/videos"))
    ap.add_argument("--out", type=Path, default=Path("audit"))
    ap.add_argument("--only", default="", help="comma-separated clip ids")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
