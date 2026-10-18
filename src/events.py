"""Turn cached per-frame signals into timed violation events (policy v7, grounded features) and score timestamps.
  python src/events.py
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ml

ROOT = Path(__file__).resolve().parent.parent
T = dict(T_client=0.2, T_pers=0.35, T_door=0.015, T_lock=0.75, bag=3)
DESK = "gemma4bg"
J = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))


def first_run(ts, flags, k=2):
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            return ts[i - k + 1]
    return None


def cached(v):
    """Feature bundle from the training caches."""
    return dict(ppl=J(f"data/people/{v}.json"), D=list(J(f"data/desk_{DESK}/{v}.json").values()),
                Q=list(J(f"data/desk_qwen4bg/{v}.json").values()), sr=J("data/screen_ratio.json")[v],
                rs=J("data/residual.json")[v], on=J("data/screen_on.json")[v],
                sf=J(f"data/safe_qwen4b/{v}.json") if (ROOT / f"data/safe_qwen4b/{v}.json").exists() else {})


def events(v=None, F=None):
    F = F or cached(v)
    ppl = F["ppl"]; ts = [float(t) for t in ppl]
    n = [sum(1 for b in bs if b[3] > 0.40) for bs in ppl.values()]
    D = (F["D"] + [F["D"][-1]] * len(ts))[:len(ts)]
    Q = (F["Q"] + [F["Q"][-1]] * len(ts))[:len(ts)]
    sr, rs, on, sf = F["sr"], F["rs"], F["on"], F["sf"]
    ev = []
    two = [x >= 2 for x in n]
    t_empty = next((ts[i] for i in range(len(n)) if n[i] == 0 and all(x == 0 for x in n[i:i + 2]) and i > 0), None)
    # moment cashier leaves her seated position (main box center moves away from its start, until empty)
    boxes = [[b for b in bs if b[3] > 0.40] for bs in ppl.values()]
    cen = [((b[0][0] + b[0][2]) / 2, (b[0][1] + b[0][3]) / 2) if len(b) == 1 else None for b in boxes]
    c0 = next((c for c in cen if c), None)
    t_leave = None
    if c0 and n[-1] == 0:
        last_seated = max([i for i, c in enumerate(cen) if c and abs(c[0] - c0[0]) + abs(c[1] - c0[1]) < 0.06] or [0])
        t_leave = ts[min(last_seated + 1, len(ts) - 1)]
        if t_empty is not None and t_empty > t_leave:
            t_leave = round((t_leave + t_empty) / 2, 1)  # walking-away moment: between leaving the seat and the room being empty
    empty_end = n[-1] == 0
    # client session
    if np.mean(two) >= T["T_client"]:
        cash = [d["cash"] or d["counting"] for d in D]
        fc = next((i for i in range(1, len(cash)) if cash[i] and cash[i - 1] and two[i] and two[i - 1]), None)
        ids = [d["id_card"] for d in D]
        visits = [i for i in range(1, len(n)) if n[i - 1] <= 1 and n[i] >= 2]
        ih = next((i for i in range(1, len(Q)) if Q[i]["id_card"] or Q[i]["cash"] or Q[i]["counting"]), None)
        t_hand = min(ts[-1], ts[ih] + 3) if ih is not None else (ts[fc] if fc is not None else ts[0])
        if fc is not None and not any(ids[:fc + 1]):
            ev.append(("231.19.1", t_hand, "Наличные приняты/пересчитываются без предъявления клиентом удостоверения личности."))
        if len(visits) >= 2:
            ev.append(("231.20.1", ts[visits[1]], "Вошёл следующий клиент, а наличные предыдущего клиента не убраны в сейф."))
        if not ev:
            ev.append(("231.19.1", t_hand, "Обслуживание клиента: возможное нарушение процедуры приёма наличных (требует проверки)."))
    # personal items
    pers = [bool(d["personal_item"]) for d in D]
    if ml.longest(pers) / len(ts) >= T["T_pers"]:
        best, cur, start, bs = 0, 0, 0, 0
        for i, p in enumerate(pers):
            cur = cur + 1 if p else 0
            if cur == 1: start = i
            if cur > best: best, bs = cur, start
        ev.append(("231.12.1", t_leave if t_leave is not None else ts[bs], "На рабочем столе кассира находятся личные вещи."))
    # door left open (scene changed vs start, outside people/laptop)
    if T["T_door"] < rs < 0.5:
        t_door = ts[min(len(ts) - 1, next((i for i, x in enumerate(n) if x >= 1), 0) + 3)]
        ev.append(("14.08.1", t_door, "Дверь кассового помещения осталась открытой после входа/выхода."))
    # cashier left with laptop unlocked / bag on desk
    if empty_end and t_empty is not None:
        if on >= 120 and not (sr is not None and sr < T["T_lock"]):
            ev.append(("53.26.1", t_leave or t_empty, "Кассир покинул рабочее место, компьютер остался разблокированным."))
        if sum(q["cash_bag"] for q in Q) >= T["bag"]:
            ev.append(("231.20.1", t_leave or t_empty, "Кассир покинул рабочее место, сумка с ценностями осталась на столе."))
    # safe open at end
    st = [sf[k] for k in sorted(sf, key=float)]
    if sum(x == "A" for x in st[-3:]) >= 2:
        i = len(st) - 1
        while i > 0 and st[i - 1] == "A":
            i -= 1
        ev.append(("231.20.1", float(sorted(sf, key=float)[i]), "Сейф оставлен открытым."))
    return ev


def main():
    gt = ml.GT
    hit2 = hit5 = typed5 = n_gt = 0; first2 = first5 = nviol = 0; errs = []
    for v in ml.VIDS:
        ev = events(v); g = gt[v]
        if g:
            nviol += 1; n_gt += len(g)
            d = [min(abs(e[1] - x["t"]) for e in ev) if ev else 99 for x in g]
            hit2 += sum(x <= 2 for x in d); hit5 += sum(x <= 5 for x in d)
            typed5 += sum(any(abs(e[1] - x["t"]) <= 5 and ml_para(e[0]) in {ml_para(r) for r in x["rules"]} for e in ev) for x in g)
            dv = min(d) if d else 99; first2 += dv <= 2; first5 += dv <= 5
            errs.append((v, [(x["t"], x["rules"][0]) for x in g], [(round(e[1], 1), e[0]) for e in ev]))
    print(f"violation videos with >=1 event within ±2s: {first2}/{nviol}, ±5s: {first5}/{nviol}")
    print(f"GT events matched by time ±2s: {hit2}/{n_gt}, ±5s: {hit5}/{n_gt}; time+rule ±5s: {typed5}/{n_gt}")
    for e in errs:
        print("  ", *e)


def ml_para(r):
    return ".".join(r.split(".")[:2])


if __name__ == "__main__":
    main()
