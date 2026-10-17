"""Video -> candidate events -> verified events -> jury result.

Pass 1 scans the whole clip in overlapping windows. Pass 2 re-checks each merged candidate on
denser frames. Timestamps and images always come from decoded frames, never from model text.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from app import checklist
from app.config import Settings
from app.media import DecodedVideo, Frame, b64
from app.rules import RuleRegistry
from app.schemas import EvidenceFrame, JobResult, Violation
from app.vlm import VLM, image_part, text_part

log = logging.getLogger("pipeline")


@dataclass
class Candidate:
    rule_id: str
    start: float
    end: float
    key_frame_id: str
    confidence: float
    observation: str
    status: str = "proposed"          # proposed | supported | not_supported | unclear
    evidence_ids: list[str] = field(default_factory=list)


def observe_schema(rule_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "enum": rule_ids},
                        "start_frame_id": {"type": "string"},
                        "end_frame_id": {"type": "string"},
                        "key_frame_id": {"type": "string"},
                        "observation": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["rule_id", "start_frame_id", "end_frame_id", "key_frame_id", "observation", "confidence"],
                },
            }
        },
        "required": ["events"],
    }


def verify_schema(rule_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["supported", "not_supported", "unclear"]},
            "rule_id": {"type": "string", "enum": rule_ids},
            "confidence": {"type": "number"},
            "evidence_frame_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
            "key_frame_id": {"type": "string"},
            "observation": {"type": "string"},
        },
        "required": ["status", "rule_id", "confidence", "evidence_frame_ids", "key_frame_id", "observation"],
    }


def frames_parts(frames: list[Frame]) -> list[dict]:
    parts = []
    for f in frames:
        parts.append(text_part(f"{f.frame_id} t={f.timestamp_sec:.2f}s"))
        parts.append(image_part(f.model_jpeg))
    return parts


def scan_windows(video: DecodedVideo, s: Settings) -> list[list[Frame]]:
    step = max(1, round(s.verify_fps / s.scan_fps))
    scan = video.frames[::step]
    duration = max(video.info.duration_sec, scan[-1].timestamp_sec)
    stride = max(1.0, s.window_sec - s.window_overlap_sec)
    windows, start = [], 0.0
    while start < duration:
        win = [f for f in scan if start <= f.timestamp_sec < start + s.window_sec]
        if win:
            windows.append(win)
        if start + s.window_sec >= duration:
            break
        start += stride
    # A tiny tail window adds cost without context; fold it into the previous one.
    if len(windows) > 1 and len(windows[-1]) < 3:
        tail = windows.pop()
        windows[-1] = sorted({f.frame_id: f for f in windows[-1] + tail}.values(), key=lambda f: f.timestamp_sec)
    return windows


def _clamp_conf(x) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


async def observe_window(vlm: VLM, window: list[Frame], video: DecodedVideo, reg: RuleRegistry, prompt: str) -> list[Candidate]:
    parts = [text_part(prompt.format(catalogue=reg.catalogue_text()))] + frames_parts(window)
    out = await vlm.chat_json(parts, observe_schema(reg.ids()), max_tokens=1200)
    allowed = {f.frame_id for f in window}
    cands = []
    for ev in out.get("events", []):
        if reg.get(ev.get("rule_id")) is None:
            continue
        ids = [ev.get("start_frame_id"), ev.get("end_frame_id"), ev.get("key_frame_id")]
        valid = [video.by_id(i) for i in ids if i in allowed]
        if not valid:
            continue  # model referenced frames it was not shown
        key = video.by_id(ev["key_frame_id"]) if ev.get("key_frame_id") in allowed else valid[0]
        times = [f.timestamp_sec for f in valid]
        cands.append(
            Candidate(
                rule_id=ev["rule_id"],
                start=min(times),
                end=max(times),
                key_frame_id=key.frame_id,
                confidence=_clamp_conf(ev.get("confidence")),
                observation=str(ev.get("observation", "")).strip(),
            )
        )
    return cands


def merge(cands: list[Candidate], gap: float) -> list[Candidate]:
    """Merge duplicates of the same rule from overlapping windows; keep distinct incidents apart."""
    merged: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.rule_id, c.start)):
        prev = merged[-1] if merged else None
        if prev and prev.rule_id == c.rule_id and c.start <= prev.end + gap:
            prev.end = max(prev.end, c.end)
            if c.confidence > prev.confidence:
                prev.key_frame_id, prev.observation, prev.confidence = c.key_frame_id, c.observation, c.confidence
        else:
            merged.append(Candidate(**{**c.__dict__, "evidence_ids": list(c.evidence_ids)}))
    return sorted(merged, key=lambda c: c.start)


def _even_subsample(frames: list[Frame], n: int) -> list[Frame]:
    if len(frames) <= n:
        return frames
    idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in sorted(set(idx))]


async def verify(vlm: VLM, c: Candidate, video: DecodedVideo, reg: RuleRegistry, prompt: str, s: Settings) -> Candidate:
    frames = _even_subsample(video.between(c.start - s.verify_pad_sec, c.end + s.verify_pad_sec), s.verify_max_frames)
    text = prompt.format(rule_id=c.rule_id, observation=c.observation, catalogue=reg.catalogue_text())
    out = await vlm.chat_json([text_part(text)] + frames_parts(frames), verify_schema(reg.ids()), max_tokens=900)
    allowed = {f.frame_id for f in frames}
    ev_ids = [i for i in out.get("evidence_frame_ids", []) if i in allowed][:3]
    key = out.get("key_frame_id") if out.get("key_frame_id") in allowed else (ev_ids[0] if ev_ids else c.key_frame_id)
    rule_id = out.get("rule_id") if reg.get(out.get("rule_id")) else c.rule_id
    return Candidate(
        rule_id=rule_id,
        start=c.start,
        end=c.end,
        key_frame_id=key,
        confidence=_clamp_conf(out.get("confidence")),
        observation=str(out.get("observation") or c.observation).strip(),
        status=out.get("status", "unclear"),
        evidence_ids=ev_ids or [key],
    )


def accepted(c: Candidate, s: Settings) -> bool:
    if c.status == "supported":
        return c.confidence >= s.accept_supported_conf
    if c.status == "checklist":  # already agreed by independent checklist votes
        return True
    if c.status == "unclear":
        return c.confidence >= s.accept_unclear_conf
    if c.status == "proposed":  # pass 2 skipped for time: keep only confident first-pass findings
        return c.confidence >= s.accept_supported_conf
    return False


async def run(video: DecodedVideo, vlm: VLM, reg: RuleRegistry, s: Settings, deadline: float) -> tuple[list[Candidate], dict]:
    observe_prompt = (s.prompts_dir / "observe.txt").read_text(encoding="utf-8")
    verify_prompt = (s.prompts_dir / "verify.txt").read_text(encoding="utf-8")
    stats: dict = {}

    t0 = time.monotonic()
    use_windows = s.pipeline_mode in ("windows", "both", "gated")
    use_checklist = s.pipeline_mode in ("checklist", "both", "gated")
    windows = scan_windows(video, s) if use_windows else []
    # Any failed call fails the job: a silent gap must never turn into "no violation".
    observed, checked_list = await asyncio.gather(
        asyncio.gather(*(observe_window(vlm, w, video, reg, observe_prompt) for w in windows)),
        checklist.run_checklist(vlm, video, reg, s) if use_checklist else _no_checklist(),
    )
    per_window = observed
    findings, cl_stats = checked_list
    stats.update(cl_stats)
    raw = [c for cs in per_window for c in cs if c.confidence >= s.candidate_min_conf]
    cands = merge(raw, s.merge_gap_sec)
    stats.update(windows=len(windows), raw_candidates=len(raw), merged_candidates=len(cands),
                 pass1_sec=round(time.monotonic() - t0, 2))

    t1 = time.monotonic()
    pass1_cost = t1 - t0
    to_verify = sorted(cands, key=lambda c: -c.confidence)[: s.max_verify_calls]
    if to_verify and time.monotonic() + max(10.0, pass1_cost) < deadline:
        checked = await asyncio.gather(*(verify(vlm, c, video, reg, verify_prompt, s) for c in to_verify))
        verified_ids = {id(c) for c in to_verify}
        unverified = [c for c in cands if id(c) not in verified_ids]
        final_pool = list(checked) + unverified
        stats["pass2_skipped"] = False
    else:
        final_pool = cands
        stats["pass2_skipped"] = bool(cands)
    stats["pass2_sec"] = round(time.monotonic() - t1, 2)

    stats["max_pass1_conf"] = round(max((c.confidence for c in raw), default=0.0), 2)
    statuses: dict[str, int] = {}
    for c in final_pool:
        statuses[c.status] = statuses.get(c.status, 0) + 1
    stats["final_status_counts"] = statuses
    # Times and statuses only (no text), for count-based diagnostics in eval_dev.py.
    stats["pass1_intervals"] = [[round(c.start, 1), round(c.end, 1), round(c.confidence, 2)] for c in raw]
    stats["final_intervals"] = [[round(c.start, 1), round(c.end, 1), c.status] for c in final_pool]
    scan_kept = [c for c in final_pool if accepted(c, s)]
    if s.pipeline_mode == "gated":
        scan_kept = [c for c in scan_kept
                     if any(f.rule_id == c.rule_id and c.start - s.gate_sec <= f.t <= c.end + s.gate_sec for f in findings)]
        stats["scan_kept_after_gate"] = len(scan_kept)
    kept = merge(scan_kept + [from_finding(f, video) for f in findings], s.merge_gap_sec)
    stats["accepted"] = len(kept)
    return kept, stats


async def _no_checklist() -> tuple[list, dict]:
    return [], {}


def from_finding(f: "checklist.Finding", video: DecodedVideo) -> Candidate:
    near = video.between(f.t - 1.0, f.t + 1.0)
    picks = [near[0], video.by_id(f.frame_id), near[-1]] if near else [video.by_id(f.frame_id)]
    ids = list(dict.fromkeys(fr.frame_id for fr in picks if fr))
    # Below typical verified confidence, so a merged scan observation (model text) wins over the template.
    return Candidate(rule_id=f.rule_id, start=f.t, end=f.t, key_frame_id=f.frame_id, confidence=0.6,
                     observation=checklist.OBS_RU[f.rule_id], status="checklist", evidence_ids=ids)


def build_result(job_id: str, events: list[Candidate], video: DecodedVideo, reg: RuleRegistry) -> JobResult:
    violations = []
    for c in sorted(events, key=lambda c: c.start):
        rule = reg.get(c.rule_id)
        key = video.by_id(c.key_frame_id)
        ids = c.evidence_ids or [c.key_frame_id]
        frames = sorted({i: video.by_id(i) for i in ids if video.by_id(i)}.values(), key=lambda f: f.timestamp_sec)[:3]
        if not frames:
            frames = [key]
        observation = c.observation.rstrip(". ") or "Зафиксировано действие, требующее проверки"
        description = (
            f"{observation}. Возможное нарушение: {rule.type_ru.lower()}. {rule.reference_ru()} "
            "Требуется проверка сотрудником."
        )
        violations.append(
            Violation(
                type=rule.type_ru,
                timecode_sec=round(key.timestamp_sec, 2),
                description=description,
                evidence_frames=[EvidenceFrame(timestamp_sec=round(f.timestamp_sec, 2), image_base64=b64(f.evidence_jpeg)) for f in frames],
            )
        )
    return JobResult(
        job_id=job_id,
        verdict="violation" if violations else "no_violation",
        violations=violations,
    )
