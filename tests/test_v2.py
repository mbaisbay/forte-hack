import asyncio
import dataclasses
import re
import time

import pytest

from app import checklist, pipeline, recheck
from app.config import Settings
from app.media import decode
from app.rules import RuleRegistry
from app.vlm import MockVLM

S = dataclasses.replace(Settings(), pipeline_mode="v2")
REG = RuleRegistry.load(S.rules_path)
NO_FLAGS = {k: {"present": "no", "frame_id": ""} for k in checklist.FLAGS}


@pytest.fixture(scope="module")
def video(synthetic_video):
    return decode(synthetic_video, S.verify_fps, S.model_max_side)


def _ids(parts):
    return [m.group(1) for p in parts if p["type"] == "text" for m in [re.match(r"^(F\d{4}) t=", p["text"])] if m]


def _is_negation(parts):
    return any(q in parts[0]["text"] for q in recheck.NEGATIONS.values())


def make(door_vote_count=2, recheck_answer="yes", scan_rule=None, conf=0.9):
    calls = {"checklist": 0, "recheck": 0}

    def responder(kind, parts):
        ids = _ids(parts)
        if kind == "checklist":
            calls["checklist"] += 1
            flags = dict(NO_FLAGS)
            if calls["checklist"] <= door_vote_count:
                flags["door_left_open"] = {"present": "yes", "frame_id": ids[len(ids) // 2]}
            return {"services": [], "cashier_leaves": [], "flags": flags}
        if kind == "recheck":
            calls["recheck"] += 1
            answer = recheck_answer
            if _is_negation(parts) and answer in ("yes", "no"):
                answer = {"yes": "no", "no": "yes"}[answer]  # a consistent model flips on the negated question
            return {"answer": answer, "frame_id": ids[len(ids) // 2], "confidence": conf, "observation": "Дверь открыта"}
        if kind == "observe" and scan_rule:
            return {"events": [{"rule_id": scan_rule, "start_frame_id": ids[2], "end_frame_id": ids[4],
                                "key_frame_id": ids[3], "observation": "x", "confidence": 0.9}]}
        return {"events": []}
    return MockVLM(responder), calls


def run(video, vlm, deadline_in=120):
    return asyncio.run(pipeline.run(video, vlm, REG, S, time.monotonic() + deadline_in))


def test_confirmed_by_recheck(video):
    vlm, calls = make()
    events, stats = run(video, vlm)
    assert [e.rule_id for e in events] == ["cash_room_door_open"]
    assert events[0].observation == "Дверь открыта" and 1 <= len(events[0].evidence_ids) <= 3
    assert calls["recheck"] == 2 and stats["candidates"] == 1  # question + its negation


def test_single_vote_is_enough_to_ask_but_unclear_never_confirms(video):
    vlm, _ = make(door_vote_count=1, recheck_answer="unclear")
    events, stats = run(video, vlm)
    assert events == [] and stats["candidates"] == 1


def test_low_confidence_answer_does_not_confirm(video):
    vlm, _ = make(conf=0.3)
    assert run(video, vlm)[0] == []


def test_scan_cannot_propose_omissions(video):
    vlm, calls = make(door_vote_count=0, scan_rule="no_client_identification")
    events, stats = run(video, vlm)
    assert events == [] and stats["candidates"] == 0 and calls["recheck"] == 0


def test_scan_visible_rule_goes_through_recheck(video):
    vlm, calls = make(door_vote_count=0, scan_rule="personal_items_in_cash_area")
    events, _ = run(video, vlm)
    assert {e.rule_id for e in events} == {"personal_items_in_cash_area"} and calls["recheck"] >= 1


def test_no_time_keeps_only_agreed_candidates(video):
    vlm, calls = make(door_vote_count=2)
    events, stats = run(video, vlm, deadline_in=0.0)
    assert stats["recheck_skipped"] and calls["recheck"] == 0
    assert [e.rule_id for e in events] == ["cash_room_door_open"]  # both checklist votes agreed
    vlm, _ = make(door_vote_count=1)
    assert run(video, vlm, deadline_in=0.0)[0] == []


def test_unreadable_rules_never_reported(video):
    def responder(kind, parts):
        ids = _ids(parts)
        if kind == "checklist":
            return {"services": [], "cashier_leaves": [{"frame_id": ids[3], "customer_present": "no",
                                                        "cash_left_on_desk": "no", "computer_locked": "no"}], "flags": NO_FLAGS}
        return {"answer": "yes", "frame_id": ids[0], "confidence": 0.9, "observation": "x"} if kind == "recheck" else {"events": []}
    assert run(video, MockVLM(responder))[0] == []


def test_same_answer_to_question_and_negation_is_not_confirmed(video):
    def responder(kind, parts):
        ids = _ids(parts)
        if kind == "checklist":
            flags = dict(NO_FLAGS); flags["door_left_open"] = {"present": "yes", "frame_id": ids[len(ids) // 2]}
            return {"services": [], "cashier_leaves": [], "flags": flags}
        if kind == "recheck":
            return {"answer": "yes", "frame_id": ids[0], "confidence": 0.9, "observation": "x"}  # biased: always "yes"
        return {"events": []}
    events, stats = run(video, MockVLM(responder))
    assert events == [] and stats["recheck_answers"][0][1] == "yes/yes"
