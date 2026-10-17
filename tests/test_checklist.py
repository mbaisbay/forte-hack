import asyncio
import re
import time

import pytest

from app import checklist, pipeline
from app.config import Settings
from app.media import decode
from app.rules import RuleRegistry
from app.vlm import MockVLM

S = Settings()
REG = RuleRegistry.load(S.rules_path)
NO_FLAGS = {k: {"present": "no", "frame_id": ""} for k in checklist.FLAGS}


@pytest.fixture(scope="module")
def video(synthetic_video):
    return decode(synthetic_video, S.verify_fps, S.model_max_side)


def _ids(parts):
    return [m.group(1) for p in parts if p["type"] == "text" for m in [re.match(r"^(F\d{4}) t=", p["text"])] if m]


def service(ids, **over):
    sv = {"start_frame_id": ids[0], "end_frame_id": ids[-1], "customer_departed": "yes",
          "id_document_checked": "yes", "id_check_frame_id": ids[1],
          "cash_first_touched_frame_id": ids[2], "amount_checked": "yes", "receipt_given": "yes", "cash_secured_after": "yes"}
    return {**sv, **over}


def test_segments_cover_clip_and_respect_frame_cap(video):
    for offset in (0, 1):
        segs = checklist.segments(video, S, offset)
        assert all(len(s) <= S.checklist_max_frames for s in segs)
        assert segs[0][0].timestamp_sec <= 1.0
        assert segs[-1][-1].timestamp_sec >= video.frames[-1].timestamp_sec - 1.0


def test_clean_service_gives_no_findings(video):
    frames = checklist.segments(video, S, 0)[0]
    ids = [f.frame_id for f in frames]
    ans = {"services": [service(ids)], "cashier_leaves": [], "flags": NO_FLAGS}
    assert checklist.findings_from(ans, frames, video) == []


def test_missing_steps_map_to_rules(video):
    frames = checklist.segments(video, S, 0)[0]
    ids = [f.frame_id for f in frames]
    ans = {"services": [service(ids, id_document_checked="no", receipt_given="no", cash_secured_after="unclear")],
           "cashier_leaves": [{"frame_id": ids[3], "customer_present": "yes", "cash_left_on_desk": "no", "computer_locked": "no"}],
           "flags": {**NO_FLAGS, "door_left_open": {"present": "yes", "frame_id": ids[4]}}}
    rules = {f.rule_id for f in checklist.findings_from(ans, frames, video)}
    # computer_locked and amount_checked are asked but not mapped (unreadable at this resolution).
    assert rules == {"no_client_identification", "no_confirmation_document", "leaves_during_service", "cash_room_door_open"}


def test_omissions_need_a_finished_service(video):
    frames = checklist.segments(video, S, 0)[0]
    ids = [f.frame_id for f in frames]
    open_service = service(ids, customer_departed="unclear", receipt_given="no", cash_secured_after="no")
    ans = {"services": [open_service], "cashier_leaves": [], "flags": NO_FLAGS}
    assert checklist.findings_from(ans, frames, video) == []
    # A following customer means the first service is over.
    ans["services"].append(service(ids))
    assert {f.rule_id for f in checklist.findings_from(ans, frames, video)} == {"no_confirmation_document", "cash_left_unsecured"}


def test_id_checked_after_cash_counts_as_violation(video):
    frames = checklist.segments(video, S, 0)[0]
    ids = [f.frame_id for f in frames]
    ans = {"services": [service(ids, id_check_frame_id=ids[-2], cash_first_touched_frame_id=ids[1])],
           "cashier_leaves": [], "flags": NO_FLAGS}
    assert [f.rule_id for f in checklist.findings_from(ans, frames, video)] == ["no_client_identification"]


def test_invented_frame_ids_are_ignored(video):
    frames = checklist.segments(video, S, 0)[0]
    ans = {"services": [], "cashier_leaves": [],
           "flags": {**NO_FLAGS, "personal_items": {"present": "yes", "frame_id": "F9999"}}}
    assert checklist.findings_from(ans, frames, video) == []


def test_votes_must_agree():
    a = [checklist.Finding("cash_room_door_open", 5.0, "F0030"), checklist.Finding("personal_items_in_cash_area", 2.0, "F0012")]
    b = [checklist.Finding("cash_room_door_open", 6.5, "F0039")]
    assert [f.rule_id for f in checklist.agree([a, b], 4.0)] == ["cash_room_door_open"]


def test_pipeline_reports_checklist_finding(video):
    def responder(kind, parts):
        if kind != "checklist":
            return {"events": []}
        ids = _ids(parts)
        return {"services": [], "cashier_leaves": [],
                "flags": {**NO_FLAGS, "door_left_open": {"present": "yes", "frame_id": ids[len(ids) // 2]}}}
    events, stats = asyncio.run(pipeline.run(video, MockVLM(responder), REG, S, time.monotonic() + 120))
    assert [e.rule_id for e in events] == ["cash_room_door_open"]
    assert stats["checklist_kept"] == 1
    result = pipeline.build_result("job", events, video, REG)
    assert result.verdict == "violation" and 1 <= len(result.violations[0].evidence_frames) <= 3


def test_gated_mode_drops_scan_events_without_checklist_support(video):
    import dataclasses
    from tests.test_pipeline import event_at
    gated = dataclasses.replace(S, pipeline_mode="gated")
    scan = event_at(4.0, rule_id="personal_items_in_cash_area")

    def no_support(kind, parts):
        if kind == "checklist":
            return {"services": [], "cashier_leaves": [], "flags": NO_FLAGS}
        return scan(kind, parts)
    events, stats = asyncio.run(pipeline.run(video, MockVLM(no_support), REG, gated, time.monotonic() + 120))
    assert events == [] and stats["scan_kept_after_gate"] == 0

    def support(kind, parts):
        if kind == "checklist":
            ids = _ids(parts)
            return {"services": [], "cashier_leaves": [],
                    "flags": {**NO_FLAGS, "personal_items": {"present": "yes", "frame_id": ids[len(ids) // 2]}}}
        return scan(kind, parts)
    events, _ = asyncio.run(pipeline.run(video, MockVLM(support), REG, gated, time.monotonic() + 120))
    assert [e.rule_id for e in events] == ["personal_items_in_cash_area"]
    assert events[0].observation == "Тестовое наблюдение"  # scan text and time win over the template
