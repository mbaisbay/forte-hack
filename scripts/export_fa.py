"""Export false-alarm / miss audit bundles for DEV clips from a finished eval run (needs LOG_CALLS=1).

Per selected clip writes <out>/<clip>/: overview frames (2 fps, <=60), dense frames (6 fps, +-2 s) around every
reported event, and case.json with labels (team + organizer), final events and every raw model answer of the job.
Holdout clips and jury uploads are never exported.

Usage (GPU box): python scripts/export_fa.py --details eval_details_dev_r14_both.json --out audit_fa
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import Settings  # noqa: E402
from app.media import decode  # noqa: E402
from eval_dev import HOLDOUT, load_labels, norm_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("open/labels_fixed.xlsx"))
    ap.add_argument("--org-labels", type=Path, default=Path("open/open_labels.xlsx"))
    ap.add_argument("--videos", type=Path, default=Path("open/videos"))
    ap.add_argument("--results", type=Path, default=Path("data/results"))
    ap.add_argument("--out", type=Path, default=Path("audit_fa"))
    args = ap.parse_args()

    s = Settings()
    team, org = load_labels(args.labels), load_labels(args.org_labels)
    details = json.loads(args.details.read_text(encoding="utf-8"))
    clips = {norm_id(p.name): p for p in args.videos.glob("*.mov")}
    picked = []
    for clip, det in details.items():
        if clip in HOLDOUT or clip not in clips:
            continue
        pred = det.get("verdict") == "violation"
        if (pred and not team[clip]["violation"]) or (not pred and team[clip]["violation"]):
            picked.append(clip)
    for clip in sorted(picked, key=lambda c: int(c.split("-")[1])):
        det = details[clip]
        video = decode(clips[clip], s.verify_fps, s.model_max_side)
        d = args.out / clip
        d.mkdir(parents=True, exist_ok=True)
        step = max(1, len(video.frames) // 60, round(s.verify_fps / 2))
        keep = {f.frame_id: f for f in video.frames[::step]}
        for e in det.get("events", []):
            keep.update({f.frame_id: f for f in video.between(e["t"] - 2, e["t"] + 2)})
        for f in sorted(keep.values(), key=lambda f: f.timestamp_sec):
            (d / f"{f.frame_id}_t{f.timestamp_sec:06.2f}.jpg").write_bytes(f.model_jpeg)
        calls_path = args.results / f"{det['job_id']}.calls.jsonl"
        calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()] if calls_path.exists() else []
        (d / "case.json").write_text(json.dumps({
            "clip": clip, "duration_sec": round(video.info.duration_sec, 2),
            "outcome": "false_alarm_on_clean_clip" if det.get("verdict") == "violation" else "missed_violation",
            "label_team": {k: team[clip][k] for k in ("violation", "events")},
            "label_organizer": {k: org[clip][k] for k in ("violation", "times")} if clip in org else None,
            "final_events": det.get("events", []),
            "raw_model_calls": calls,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"exported {clip}: {len(keep)} frames, {len(calls)} calls", flush=True)


if __name__ == "__main__":
    main()
