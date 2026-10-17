"""Focused re-check: one narrow question per rule, asked on dense frames around the candidate moment.

The audit showed the model fails on small objects (cards, receipts, banknotes) when it sees many
downscaled frames at once. Here it sees a short window at 6 fps, by default at the camera's native
resolution, and answers one yes/no question. A candidate becomes a violation only if the answer is
the violating one; "unclear" never confirms.
"""

import asyncio
from dataclasses import dataclass

from app.config import Settings
from app.media import DecodedVideo, Frame
from app.vlm import VLM, image_part, text_part


@dataclass(frozen=True)
class Spec:
    question: str
    violation: str      # the answer that confirms the violation: "yes" or "no"
    before: float       # seconds of context before the candidate start
    after: float        # seconds after the candidate end


OPPOSITE = {"yes": "no", "no": "yes"}
# Opposite-polarity questions: an 8B model answers "no" to most yes/no questions (r9: 21/21 "no" on cash-left
# candidates, 7/7 "no" = confirmation on clean-clip ID candidates). Requiring consistent answers to a question and
# its negation cancels that bias.
NEGATIONS = {
    "no_client_identification": "Does the cashier take or count the customer's cash before any ID card, passport or "
        "phone screen has been handed over or shown?",
    "no_confirmation_document": "Does the customer leave the window without receiving any paper, receipt or document "
        "from the cashier?",
    "cash_left_unsecured": "Is all cash on the desk put away into a drawer, safe or cabinet (and closed), or handed to "
        "the customer, before a new customer is served or the cashier turns to another task?",
    "leaves_during_service": "Does the cashier stay at the workstation for as long as the customer is at the window?",
    "cash_room_door_open": "Is the door of the cashier's booth or cash room closed, or only opened briefly to pass, "
        "throughout these frames?",
    "outsider_in_cash_room": "Is the cashier the only person inside the cashier's enclosed working area (nobody else "
        "behind the counter)?",
    "personal_items_in_cash_area": "Is the cashier's desk free of personal items (handbag, wallet, personal phone, cup, "
        "food or drink)?",
    "denominations_mixed": "Is each stack or band made of banknotes of one denomination only (same colour and size)?",
    "documents_cash_mixed": "Does the cashier count the cash for each document separately?",
    "collection_bag_not_checked": "Does the cashier turn over and inspect the collection bag and its seal before "
        "accepting or opening it?",
    "collection_visible_to_outsiders": "Does the cash collection handover take place out of sight and reach of "
        "customers and outsiders?",
}


SPECS: dict[str, Spec] = {
    "no_client_identification": Spec(
        "Before the cashier first takes or counts the customer's cash, does the customer hand over or show an ID card, "
        "passport or a phone screen, and does the cashier take it or look at it?", "no", 10.0, 3.0),
    "no_confirmation_document": Spec(
        "Before the customer leaves the window, does the cashier hand the customer a paper, receipt or document?",
        "no", 10.0, 2.0),
    "cash_left_unsecured": Spec(
        "Does cash stay lying on the cashier's desk while a new customer is served, the cashier turns to another task or "
        "leaves, instead of being put into a drawer, safe or cabinet or handed to the customer? Also answer yes if a safe "
        "or cash drawer is left open and unattended.", "yes", 3.0, 8.0),
    "leaves_during_service": Spec(
        "Does the cashier get up and leave the workstation while a customer is still at the window?", "yes", 3.0, 6.0),
    "cash_room_door_open": Spec(
        "Does the door of the cashier's booth or cash room stay open for at least 4 seconds in these frames "
        "(not only opened briefly to pass and closed again)?", "yes", 2.0, 8.0),
    "outsider_in_cash_room": Spec(
        "Is a person who is not the cashier inside the cashier's enclosed working area (behind the counter, next to the "
        "cashier's desk or safe), not on the customer side of the window?", "yes", 2.0, 5.0),
    "personal_items_in_cash_area": Spec(
        "Is there a personal item on or at the cashier's desk: a handbag, wallet, personal mobile phone, cup, food or "
        "drink? A computer, keyboard, counting machine, stamp, documents or desk phone do not count.", "yes", 2.0, 3.0),
    "denominations_mixed": Spec(
        "Does the cashier put banknotes of visibly different colours or sizes (different denominations) together into "
        "one stack or band?", "yes", 3.0, 5.0),
    "documents_cash_mixed": Spec(
        "Does the cashier count cash that belongs to several separate documents together as one pile instead of "
        "separately?", "yes", 5.0, 5.0),
    "collection_bag_not_checked": Spec(
        "Is a cash collection bag accepted or opened without the cashier first turning it over and inspecting the bag "
        "and its seal?", "yes", 5.0, 5.0),
    "collection_visible_to_outsiders": Spec(
        "Does a cash collection handover (bags, cases, collectors) take place where customers or other outsiders can see "
        "it or come close to it?", "yes", 3.0, 5.0),
}

# Rules the model cannot judge at this resolution (audit 2026-10-17): never reported.
UNREADABLE = {"amount_not_reconciled", "workstation_not_locked"}
# Scan pass proposes only rules that are visible states or actions; omissions come from the checklist.
SCAN_RULES = {"cash_left_unsecured", "leaves_during_service", "cash_room_door_open", "outsider_in_cash_room",
              "personal_items_in_cash_area", "denominations_mixed", "documents_cash_mixed",
              "collection_bag_not_checked", "collection_visible_to_outsiders"}


def schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no", "unclear"]},
            "frame_id": {"type": "string"},
            "confidence": {"type": "number"},
            "observation": {"type": "string"},
        },
        "required": ["answer", "frame_id", "confidence", "observation"],
    }


def window_frames(video: DecodedVideo, start: float, end: float, spec: Spec, s: Settings) -> list[Frame]:
    frames = video.between(start - spec.before, end + spec.after)
    if len(frames) <= s.recheck_max_frames:
        return frames
    idx = [round(i * (len(frames) - 1) / (s.recheck_max_frames - 1)) for i in range(s.recheck_max_frames)]
    return [frames[i] for i in sorted(set(idx))]


def parts(frames: list[Frame], native: bool) -> list[dict]:
    out = []
    for f in frames:
        out.append(text_part(f"{f.frame_id} t={f.timestamp_sec:.2f}s"))
        out.append(image_part(f.evidence_jpeg if native else f.model_jpeg))
    return out


async def ask(vlm: VLM, rule_id: str, start: float, end: float, video: DecodedVideo, prompt: str, s: Settings) -> dict | None:
    """Returns {"confirmed", "frame_id", "confidence", "observation", "answer"} or None when the rule has no spec."""
    spec = SPECS.get(rule_id)
    if spec is None:
        return None
    frames = window_frames(video, start, end, spec, s)
    if not frames:
        return None
    images = parts(frames, s.recheck_native)

    async def one(question: str) -> tuple[str, float, dict]:
        out = await vlm.chat_json([text_part(prompt.format(question=question))] + images, schema(), max_tokens=500)
        try:
            conf = min(1.0, max(0.0, float(out.get("confidence", 0))))
        except (TypeError, ValueError):
            conf = 0.0
        return out.get("answer", "unclear"), conf, out

    negation = NEGATIONS.get(rule_id) if s.recheck_contrast else None
    if negation:
        (answer, conf, out), (neg_answer, neg_conf, _) = await asyncio.gather(one(spec.question), one(negation))
        consistent = neg_answer == OPPOSITE.get(spec.violation) and neg_conf >= s.recheck_min_conf
        label = f"{answer}/{neg_answer}"
    else:
        answer, conf, out = await one(spec.question)
        consistent, label = True, answer
    allowed = {f.frame_id for f in frames}
    return {
        "answer": label,
        "confirmed": answer == spec.violation and conf >= s.recheck_min_conf and consistent,
        "frame_id": out.get("frame_id") if out.get("frame_id") in allowed else None,
        "confidence": conf,
        "observation": str(out.get("observation", "")).strip(),
    }
