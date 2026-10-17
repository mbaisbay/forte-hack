"""Scene-state timeline (the team plan's "timeline + debounce rules"), limited to states the 8B model reads well.

One pass samples the clip every ~1.5 s and asks for five scene states per frame (door, safe, cash on desk,
cashier at desk, customer at desk). Plain code then applies persistence rules, so a door that is open for one
frame or a cashier who stands up for a moment never becomes a violation. "unknown" breaks every run.
"""

import asyncio
from dataclasses import dataclass

from app.config import Settings
from app.media import DecodedVideo, Frame
from app.vlm import VLM, text_part

STATE = {"type": "string", "enum": ["yes", "no", "unknown"]}
FIELDS = ["door_open", "safe_open", "cash_on_desk", "cashier_at_desk", "customer_at_desk"]
# Rules this pass decides; in "states" mode their scan/checklist events are replaced by these.
STATE_RULES = {"cash_room_door_open", "leaves_during_service", "cash_left_unsecured"}

OBS_RU = {
    "cash_room_door_open": "Дверь кассового помещения остаётся открытой {d:.0f} с",
    "leaves_during_service": "Кассир отсутствует на рабочем месте {d:.0f} с, пока клиент у кассы",
    "cash_left_unsecured": "Наличные остаются на столе {why}",
}


def schema() -> dict:
    row = {"type": "object", "properties": {"frame_id": {"type": "string"}, **{f: STATE for f in FIELDS}},
           "required": ["frame_id", *FIELDS]}
    return {"type": "object", "properties": {"rows": {"type": "array", "items": row}}, "required": ["rows"]}


@dataclass
class Row:
    t: float
    frame_id: str
    s: dict


@dataclass
class StateEvent:
    rule_id: str
    t: float
    frame_id: str
    observation: str


def sample(video: DecodedVideo, s: Settings) -> list[list[Frame]]:
    step = max(1, round(s.verify_fps * s.state_period_sec))
    frames = video.frames[::step]
    n = s.state_max_frames
    return [frames[i:i + n] for i in range(0, len(frames), n)]


async def timeline(vlm: VLM, video: DecodedVideo, prompt: str, s: Settings) -> list[Row]:
    from app.pipeline import frames_parts  # local import: pipeline imports this module

    async def seg(frames: list[Frame]) -> list[Row]:
        out = await vlm.chat_json([text_part(prompt)] + frames_parts(frames), schema(),
                                  max_tokens=60 * len(frames) + 200)
        by_id = {f.frame_id: f for f in frames}
        rows = []
        for r in out.get("rows", []):
            f = by_id.get(r.get("frame_id"))
            if f:
                rows.append(Row(f.timestamp_sec, f.frame_id, {k: r.get(k, "unknown") for k in FIELDS}))
        return rows

    parts = await asyncio.gather(*(seg(fr) for fr in sample(video, s)))
    rows = sorted({r.frame_id: r for rs in parts for r in rs}.values(), key=lambda r: r.t)
    return rows


def runs(rows: list[Row], pred) -> list[list[Row]]:
    """Maximal runs of consecutive rows where pred(row) is True; a row failing pred (incl. unknown) ends a run."""
    out, cur = [], []
    for r in rows:
        if pred(r):
            cur.append(r)
        else:
            if cur:
                out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def span(run: list[Row], period: float) -> float:
    # A run of n samples covers about n sampling periods.
    return run[-1].t - run[0].t + period


def events_from(rows: list[Row], s: Settings) -> list[StateEvent]:
    p = s.state_period_sec
    ev: list[StateEvent] = []
    for run in runs(rows, lambda r: r.s["door_open"] == "yes"):
        d = span(run, p)
        if d >= s.state_door_min_sec:
            ev.append(StateEvent("cash_room_door_open", run[0].t, run[0].frame_id, OBS_RU["cash_room_door_open"].format(d=d)))
    for run in runs(rows, lambda r: r.s["cashier_at_desk"] == "no" and r.s["customer_at_desk"] == "yes"):
        d = span(run, p)
        if d >= s.state_away_min_sec:
            ev.append(StateEvent("leaves_during_service", run[0].t, run[0].frame_id, OBS_RU["leaves_during_service"].format(d=d)))
    for run in runs(rows, lambda r: r.s["cash_on_desk"] == "yes"):
        # Persistent cash: still on the desk when a new customer arrives, or while the cashier is away.
        for a, b in zip(run, run[1:]):
            if a.s["customer_at_desk"] == "no" and b.s["customer_at_desk"] == "yes" and run[0].t < a.t:
                ev.append(StateEvent("cash_left_unsecured", b.t, b.frame_id,
                                     OBS_RU["cash_left_unsecured"].format(why="при подходе следующего клиента")))
                break
        away = runs(run, lambda r: r.s["cashier_at_desk"] == "no")
        long_away = [w for w in away if span(w, p) >= s.state_away_min_sec]
        if long_away:
            w = long_away[0]
            ev.append(StateEvent("cash_left_unsecured", w[0].t, w[0].frame_id,
                                 OBS_RU["cash_left_unsecured"].format(why="во время отсутствия кассира")))
    for run in runs(rows, lambda r: r.s["safe_open"] == "yes" and r.s["cashier_at_desk"] == "no"):
        if span(run, p) >= s.state_away_min_sec:
            ev.append(StateEvent("cash_left_unsecured", run[0].t, run[0].frame_id,
                                 "Сейф или денежный ящик оставлен открытым без присмотра"))
    return ev
