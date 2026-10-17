"""Whole-clip checklist pass: the model fills a fixed procedure checklist, code maps answers to rules.

Windowed scanning cannot see a step that never happens (no ID check, no receipt). Here the model
answers fixed questions over the whole service, and plain code decides which rule fires. Two calls on
offset frame grids must agree before a finding is kept.
"""

import asyncio
from dataclasses import dataclass

from app.config import Settings
from app.media import DecodedVideo, Frame
from app.rules import RuleRegistry
from app.vlm import VLM, text_part

TRI = {"type": "string", "enum": ["yes", "no", "unclear"]}
FID = {"type": "string"}
FLAG = {"type": "object", "properties": {"present": TRI, "frame_id": FID}, "required": ["present", "frame_id"]}
FLAGS = {
    "door_left_open": "cash_room_door_open",
    "safe_left_open": "cash_left_unsecured",
    "outsider_inside": "outsider_in_cash_room",
    "personal_items": "personal_items_in_cash_area",
    "denominations_mixed": "denominations_mixed",
    "several_documents_mixed": "documents_cash_mixed",
    "collection_bag_unchecked": "collection_bag_not_checked",
}

# Code-written observations: the checklist answers are yes/no, so the wording is ours, not the model's.
OBS_RU = {
    "no_client_identification": "Кассир начал работу с наличными клиента без видимой проверки документа, удостоверяющего личность",
    "amount_not_reconciled": "После пересчёта не видно сверки суммы с документом или экраном",
    "no_confirmation_document": "Клиент отошёл от кассы, не получив документ, подтверждающий операцию",
    "cash_left_unsecured": "Наличные остались на рабочем месте вне закрываемого хранилища",
    "leaves_during_service": "Кассир покинул рабочее место, пока клиент ещё у кассы",
    "workstation_not_locked": "Кассир покинул рабочее место, не заблокировав компьютер",
    "cash_room_door_open": "Дверь кассового помещения оставлена открытой",
    "outsider_in_cash_room": "В кассовой зоне находится посторонний человек",
    "personal_items_in_cash_area": "На рабочем месте кассира находятся личные вещи",
    "denominations_mixed": "Банкноты разных номиналов сложены вместе",
    "documents_cash_mixed": "Наличные по нескольким документам пересчитаны вместе",
    "collection_bag_not_checked": "Инкассаторская сумка принята без видимой проверки целостности и пломбы",
}


def checklist_schema() -> dict:
    service = {
        "type": "object",
        "properties": {
            "start_frame_id": FID, "end_frame_id": FID,
            "id_document_checked": TRI, "id_check_frame_id": FID,
            "cash_first_touched_frame_id": FID,
            "amount_checked": TRI, "receipt_given": TRI, "cash_secured_after": TRI,
        },
        "required": ["start_frame_id", "end_frame_id", "id_document_checked", "id_check_frame_id",
                     "cash_first_touched_frame_id", "amount_checked", "receipt_given", "cash_secured_after"],
    }
    leave = {
        "type": "object",
        "properties": {"frame_id": FID, "customer_present": TRI, "cash_left_on_desk": TRI, "computer_locked": TRI},
        "required": ["frame_id", "customer_present", "cash_left_on_desk", "computer_locked"],
    }
    return {
        "type": "object",
        "properties": {
            "services": {"type": "array", "maxItems": 4, "items": service},
            "cashier_leaves": {"type": "array", "maxItems": 3, "items": leave},
            "flags": {"type": "object", "properties": {k: FLAG for k in FLAGS}, "required": list(FLAGS)},
        },
        "required": ["services", "cashier_leaves", "flags"],
    }


@dataclass
class Finding:
    rule_id: str
    t: float
    frame_id: str


def segments(video: DecodedVideo, s: Settings, offset: int) -> list[list[Frame]]:
    """Whole clip, or overlapping long segments, sampled on a grid shifted by `offset` source frames."""
    seg, overlap = s.checklist_segment_sec, 10.0
    duration = video.frames[-1].timestamp_sec
    out, start = [], 0.0
    while True:
        frames = [f for f in video.between(start, start + seg)]
        step = max(1, -(-len(frames) // s.checklist_max_frames), round(s.verify_fps / s.checklist_fps))
        picked = frames[min(offset, len(frames) - 1)::step][: s.checklist_max_frames] if frames else []
        if picked:
            out.append(picked)
        if start + seg >= duration:
            return out
        start += seg - overlap


def findings_from(answer: dict, frames: list[Frame], video: DecodedVideo) -> list[Finding]:
    shown = {f.frame_id: f for f in frames}
    at = lambda fid: shown.get(fid)
    out: list[Finding] = []

    def add(rule_id: str, *fids: str) -> None:
        for fid in fids:
            if at(fid):
                out.append(Finding(rule_id, at(fid).timestamp_sec, fid))
                return

    for sv in answer.get("services", []):
        cash, idf, end = sv.get("cash_first_touched_frame_id"), sv.get("id_check_frame_id"), sv.get("end_frame_id")
        if at(cash):
            late_id = at(idf) and at(idf).timestamp_sec > at(cash).timestamp_sec + 1.0
            if sv.get("id_document_checked") == "no" or late_id:
                add("no_client_identification", cash)
            if sv.get("amount_checked") == "no":
                add("amount_not_reconciled", end, cash)
            if sv.get("cash_secured_after") == "no":
                add("cash_left_unsecured", end, cash)
        if sv.get("receipt_given") == "no":
            add("no_confirmation_document", end)
    for lv in answer.get("cashier_leaves", []):
        fid = lv.get("frame_id")
        if lv.get("customer_present") == "yes":
            add("leaves_during_service", fid)
        if lv.get("cash_left_on_desk") == "yes":
            add("cash_left_unsecured", fid)
        if lv.get("computer_locked") == "no":
            add("workstation_not_locked", fid)
    for key, rule_id in FLAGS.items():
        flag = answer.get("flags", {}).get(key, {})
        if flag.get("present") == "yes":
            add(rule_id, flag.get("frame_id", ""))
    return out


async def one_vote(vlm: VLM, video: DecodedVideo, prompt: str, s: Settings, offset: int) -> list[Finding]:
    from app.pipeline import frames_parts  # local import: pipeline imports this module

    async def seg_call(frames: list[Frame]) -> list[Finding]:
        out = await vlm.chat_json([text_part(prompt)] + frames_parts(frames), checklist_schema(), max_tokens=1500)
        return findings_from(out, frames, video)

    per_seg = await asyncio.gather(*(seg_call(f) for f in segments(video, s, offset)))
    return [x for xs in per_seg for x in xs]


def agree(votes: list[list[Finding]], tol: float) -> list[Finding]:
    """Keep findings of the first vote that every other vote also reports (same rule, close in time)."""
    if not votes:
        return []
    kept = []
    for f in votes[0]:
        if all(any(g.rule_id == f.rule_id and abs(g.t - f.t) <= tol for g in other) for other in votes[1:]):
            kept.append(f)
    return kept


async def run_checklist(vlm: VLM, video: DecodedVideo, reg: RuleRegistry, s: Settings) -> tuple[list[Finding], dict]:
    prompt = (s.prompts_dir / "checklist.txt").read_text(encoding="utf-8")
    step = max(1, round(s.verify_fps / s.checklist_fps))
    offsets = [round(i * step / s.checklist_votes) for i in range(s.checklist_votes)]
    votes = await asyncio.gather(*(one_vote(vlm, video, prompt, s, o) for o in offsets))
    kept = [f for f in agree(list(votes), s.checklist_agree_sec) if reg.get(f.rule_id)]
    stats = {
        "checklist_raw": [len(v) for v in votes],
        "checklist_kept": len(kept),
        "checklist_rules": sorted({f.rule_id for f in kept}),
    }
    return kept, stats
