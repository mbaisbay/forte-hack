import asyncio
import base64
import re
import time

import pytest

from app import pipeline
from app.config import Settings
from app.media import decode
from app.rules import RuleRegistry
from app.vlm import MockVLM, VLMError

S = Settings()
REG = RuleRegistry.load(S.rules_path)


def _frame_ids(parts):
    return [m.group(1) for p in parts if p["type"] == "text" for m in [re.match(r"^(F\d{4}) t=", p["text"])] if m]


def _ts(parts):
    return {m.group(1): float(m.group(2)) for p in parts if p["type"] == "text"
            for m in [re.match(r"^(F\d{4}) t=([\d.]+)s", p["text"])] if m}


def event_at(second: float, rule_id="leaves_during_service", conf=0.9, status="supported"):
    """Mock that 'sees' one event at a fixed time, in every window and verify call covering it."""
    def responder(kind, parts):
        ts = _ts(parts)
        near = [fid for fid, t in ts.items() if abs(t - second) <= 0.5]
        if kind == "observe":
            if not near:
                return {"events": []}
            return {"events": [{"rule_id": rule_id, "start_frame_id": near[0], "end_frame_id": near[-1],
                                "key_frame_id": near[0], "observation": "Тестовое наблюдение", "confidence": conf}]}
        return {"status": status, "rule_id": rule_id, "confidence": conf,
                "evidence_frame_ids": near[:2] or list(ts)[:1], "key_frame_id": (near or list(ts))[0],
                "observation": "Тестовое наблюдение"}
    return responder


@pytest.fixture(scope="module")
def video(synthetic_video):
    return decode(synthetic_video, S.verify_fps, S.model_max_side)


def run(video, vlm, deadline_in=120):
    return asyncio.run(pipeline.run(video, vlm, REG, S, time.monotonic() + deadline_in))


def test_windows_cover_whole_clip(video):
    windows = pipeline.scan_windows(video, S)
    covered = sorted({f.timestamp_sec for w in windows for f in w})
    assert covered[0] == 0.0
    assert covered[-1] >= video.info.duration_sec - 1.0


def test_no_events_gives_no_violation(video):
    events, _ = run(video, MockVLM())
    result = pipeline.build_result("j1", events, video, REG)
    assert result.verdict == "no_violation" and result.violations == []


def test_event_in_overlap_is_reported_once(video):
    # 9 s falls inside the overlap of window [0,10) and [8,18)
    vlm = MockVLM(event_at(9.0))
    events, stats = run(video, vlm)
    assert stats["raw_candidates"] == 2 and stats["merged_candidates"] == 1
    result = pipeline.build_result("j2", events, video, REG)
    assert result.verdict == "violation" and len(result.violations) == 1
    v = result.violations[0]
    assert abs(v.timecode_sec - 9.0) <= 0.5
    assert 1 <= len(v.evidence_frames) <= 3
    for ef in v.evidence_frames:
        assert base64.b64decode(ef.image_base64)[:2] == b"\xff\xd8"
    assert REG.get("leaves_during_service").type_ru == v.type


def test_not_supported_is_dropped(video):
    events, _ = run(video, MockVLM(event_at(5.0, status="not_supported")))
    assert events == []


def test_hallucinated_frame_ids_are_ignored(video):
    def responder(kind, parts):
        return {"events": [{"rule_id": "leaves_during_service", "start_frame_id": "F9999", "end_frame_id": "F9999",
                            "key_frame_id": "F9999", "observation": "x", "confidence": 0.99}]}
    events, stats = run(video, MockVLM(responder))
    assert stats["raw_candidates"] == 0 and events == []


def test_vlm_failure_raises_not_negative(video):
    class Broken:
        async def chat_json(self, *a, **k):
            raise VLMError("down")
    with pytest.raises(VLMError):
        run(video, Broken())


def test_pass2_skipped_near_deadline_keeps_confident(video):
    vlm = MockVLM(event_at(3.0, conf=0.9))
    events, stats = run(video, vlm, deadline_in=0)
    assert stats["pass2_skipped"] is True and "verify" not in vlm.calls
    assert len(events) == 1


def test_two_distinct_incidents_stay_separate(long_synthetic_video):
    video = decode(long_synthetic_video, S.verify_fps, S.model_max_side)
    a, b = event_at(4.0), event_at(25.0)
    def responder(kind, parts):
        ra, rb = a(kind, parts), b(kind, parts)
        if kind == "observe":
            return {"events": ra["events"] + rb["events"]}
        ts = list(_ts(parts).values())
        return ra if min(ts) < 15 else rb
    events, _ = run(video, MockVLM(responder))
    assert len(events) == 2


def test_local_vlm_works_across_event_loops(monkeypatch):
    """Each job runs asyncio.run(); the client must not reuse a semaphore from an old loop."""
    import httpx
    from app.vlm import LocalVLM

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": '{"events": []}'}, "finish_reason": "stop"}]}

    async def fake_post(self, url, json):
        return FakeResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    vlm = LocalVLM("http://127.0.0.1:8000/v1", "m", concurrency=1, timeout_sec=5)
    for _ in range(3):
        async def many():
            return await asyncio.gather(*(vlm.chat_json([], {}, 10) for _ in range(4)))
        assert len(asyncio.run(many())) == 4


def test_local_vlm_rejects_remote_url():
    from app.vlm import LocalVLM
    with pytest.raises(VLMError):
        LocalVLM("https://api.example.com/v1", "m", 1, 5)


def test_merge_keeps_evidence_with_key_frame():
    a = pipeline.Candidate("leaves_during_service", 1.0, 2.0, "F0006", 0.6, "a", "supported", ["F0004", "F0006"])
    b = pipeline.Candidate("leaves_during_service", 2.5, 3.0, "F0016", 0.9, "b", "supported", ["F0015", "F0016", "F0017"])
    m = pipeline.merge([a, b], 3.0)
    assert len(m) == 1 and m[0].key_frame_id == "F0016" and m[0].evidence_ids == ["F0015", "F0016", "F0017"]


def test_result_evidence_contains_key_frame(video):
    c = pipeline.Candidate("leaves_during_service", 4.0, 4.0, video.frames[30].frame_id, 0.9, "x", "supported",
                           [video.frames[1].frame_id, video.frames[2].frame_id, video.frames[3].frame_id])
    r = pipeline.build_result("job", [c], video, REG)
    ts = [f.timestamp_sec for f in r.violations[0].evidence_frames]
    assert round(video.frames[30].timestamp_sec, 2) in ts and len(ts) <= 3


def test_semaphore_is_per_loop_across_threads():
    import threading
    from app.vlm import LocalVLM
    vlm = LocalVLM("http://127.0.0.1:9/v1", "m", 2, 1.0)
    seen, errors = [], []

    def job():
        async def main():
            a, b = vlm._sem(), vlm._sem()
            seen.append(a is b)
        try:
            asyncio.run(main())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    ts = [threading.Thread(target=job) for _ in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert not errors and all(seen) and len(seen) == 8


def test_raw_calls_are_logged_per_job(video, tmp_path, monkeypatch):
    import json as _json
    import httpx
    from app import vlm as vlm_mod

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": '{"events": []}'}, "finish_reason": "stop"}]}

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    client = vlm_mod.LocalVLM("http://127.0.0.1:8000/v1", "m", 2, 5.0)
    path = tmp_path / "job.calls.jsonl"
    token = vlm_mod.CALL_LOG.set(path)
    try:
        parts = pipeline.frames_parts(video.frames[:3])
        asyncio.run(client.chat_json(parts, pipeline.observe_schema(REG.ids()), max_tokens=50))
    finally:
        vlm_mod.CALL_LOG.reset(token)
    rec = _json.loads(path.read_text().splitlines()[0])
    assert rec["kind"] == "observe" and rec["frames"] == [f.frame_id for f in video.frames[:3]]
    assert rec["content"] == '{"events": []}' and "image" not in path.read_text()
