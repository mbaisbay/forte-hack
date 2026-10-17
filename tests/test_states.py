import asyncio
import dataclasses
import re
import time

import pytest

from app import checklist, pipeline, states
from app.config import Settings
from app.media import decode
from app.rules import RuleRegistry
from app.vlm import MockVLM

S = dataclasses.replace(Settings(), state_timeline=True)
REG = RuleRegistry.load(S.rules_path)


def rows(spec: list[str], period=1.5):
    """spec items like 'D-k-u': door, safe, cash, cashier, customer as Y/N/?."""
    m = {"Y": "yes", "N": "no", "?": "unknown"}
    out = []
    for i, code in enumerate(spec):
        d, sf, c, k, u = code
        out.append(states.Row(i * period, f"F{i:04d}", dict(zip(states.FIELDS, (m[d], m[sf], m[c], m[k], m[u])))))
    return out


def rules(evs):
    return [e.rule_id for e in evs]


def test_brief_door_opening_is_not_a_violation():
    assert rules(states.events_from(rows(["NNNYN", "YNNYN", "NNNYN"]), S)) == []


def test_door_open_four_seconds_is_reported_at_onset():
    ev = states.events_from(rows(["NNNYN", "YNNYN", "YNNYN", "YNNYN", "NNNYN"]), S)
    assert rules(ev) == ["cash_room_door_open"] and ev[0].t == 1.5


def test_unknown_breaks_a_run():
    assert rules(states.events_from(rows(["YNNYN", "?NNYN", "YNNYN", "?NNYN"]), S)) == []


def test_cashier_away_while_customer_waits():
    ev = states.events_from(rows(["NNNYY", "NNNNY", "NNNNY", "NNNYY"]), S)
    assert rules(ev) == ["leaves_during_service"]
    # Away with nobody at the desk is fine.
    assert rules(states.events_from(rows(["NNNYN", "NNNNN", "NNNNN", "NNNYN"]), S)) == []


def test_cash_still_out_when_next_customer_arrives():
    ev = states.events_from(rows(["NNYYY", "NNYYN", "NNYYN", "NNYYY"]), S)
    assert rules(ev) == ["cash_left_unsecured"] and ev[0].t == 4.5
    # Cash put away before the next customer: no event.
    assert rules(states.events_from(rows(["NNYYY", "NNNYN", "NNNYY"]), S)) == []


def test_safe_open_while_cashier_away():
    assert rules(states.events_from(rows(["NYNYN", "NYNNN", "NYNNN", "NNNYN"]), S)) == ["cash_left_unsecured"]


@pytest.fixture(scope="module")
def video(synthetic_video):
    return decode(synthetic_video, S.verify_fps, S.model_max_side)


def test_state_events_replace_scan_events_for_state_rules(video):
    flags = {k: {"present": "no", "frame_id": ""} for k in checklist.FLAGS}

    def responder(kind, parts):
        ids = [m.group(1) for p in parts if p["type"] == "text" for m in [re.match(r"^(F\d{4}) t=", p["text"])] if m]
        if kind == "states":
            door = lambda i: "yes" if 2 <= i <= 6 else "no"
            return {"rows": [{"frame_id": f, "door_open": door(i), "safe_open": "no", "cash_on_desk": "no",
                              "cashier_at_desk": "yes", "customer_at_desk": "no"} for i, f in enumerate(ids)]}
        if kind == "checklist":
            return {"services": [], "cashier_leaves": [], "flags": flags}
        if kind == "verify":
            return {"status": "supported", "rule_id": "leaves_during_service", "confidence": 0.9,
                    "evidence_frame_ids": ids[:1], "key_frame_id": ids[0], "observation": "x"}
        # The scan claims the cashier left: must be dropped in favour of the state timeline.
        return {"events": [{"rule_id": "leaves_during_service", "start_frame_id": ids[1], "end_frame_id": ids[2],
                            "key_frame_id": ids[1], "observation": "x", "confidence": 0.9}]}
    events, stats = asyncio.run(pipeline.run(video, MockVLM(responder), REG, S, time.monotonic() + 120))
    assert [e.rule_id for e in events] == ["cash_room_door_open"]
    assert stats["state_rows"] > 0
