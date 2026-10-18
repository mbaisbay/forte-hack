"""Decision tree (data/tree.json) over per-frame signals -> timed violation events.
Same logic as policy v7; thresholds, on/off switches, rule mapping and messages come from the tree."""
import numpy as np


def longest(x):
    best = cur = 0
    for b in x:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def events(F, tree):
    """F: signals from detect.py. Returns [(node_id, t, message, rule_ids)]."""
    N = {n["id"]: n for n in tree["nodes"]}
    on = lambda k: N.get(k, {}).get("enabled", False) and (N[k]["parent"] is None or on(N[k]["parent"]))
    P = lambda k, p: N[k]["params"][p]
    G = tree["params"]
    ev = []
    add = lambda k, t: ev.append((k, float(t), N[k]["message"], N[k]["rule_ids"]))

    ppl = F["ppl"]; ts = [float(t) for t in ppl]
    boxes = [[b for b in bs if b[3] > G["person_min_y"]] for bs in ppl.values()]
    n = [len(b) for b in boxes]
    D = (F["D"] + [F["D"][-1]] * len(ts))[:len(ts)]
    Q = (F["Q"] + [F["Q"][-1]] * len(ts))[:len(ts)]
    sr, rs, scr, sf = F["sr"], F["rs"], F["on"], F["sf"]
    two = [x >= 2 for x in n]
    t_empty = next((ts[i] for i in range(len(n)) if n[i] == 0 and all(x == 0 for x in n[i:i + 2]) and i > 0), None)
    # moment the cashier leaves the seated position (main box centre moves away from its start, until empty)
    cen = [((b[0][0] + b[0][2]) / 2, (b[0][1] + b[0][3]) / 2) if len(b) == 1 else None for b in boxes]
    c0 = next((c for c in cen if c), None)
    t_leave = None
    if c0 and n[-1] == 0:
        last_seated = max([i for i, c in enumerate(cen) if c and abs(c[0] - c0[0]) + abs(c[1] - c0[1]) < 0.06] or [0])
        t_leave = ts[min(last_seated + 1, len(ts) - 1)]
        if t_empty is not None and t_empty > t_leave:
            t_leave = round((t_leave + t_empty) / 2, 1)

    if on("client_session") and np.mean(two) >= P("client_session", "T_client"):
        cash = [d["cash"] or d["counting"] for d in D]
        fc = next((i for i in range(1, len(cash)) if cash[i] and cash[i - 1] and two[i] and two[i - 1]), None)
        ids = [d["id_card"] for d in D]
        visits = [i for i in range(1, len(n)) if n[i - 1] <= 1 and n[i] >= 2]
        ih = next((i for i in range(1, len(Q)) if Q[i]["id_card"] or Q[i]["cash"] or Q[i]["counting"]), None)
        t_hand = min(ts[-1], ts[ih] + 3) if ih is not None else (ts[fc] if fc is not None else ts[0])
        k0 = len(ev)
        if on("no_id") and fc is not None and not any(ids[:fc + 1]):
            add("no_id", t_hand)
        if on("next_client") and len(visits) >= 2:
            add("next_client", ts[visits[1]])
        if on("client_fallback") and len(ev) == k0:
            add("client_fallback", t_hand)

    if on("solo_cash") and not (np.mean(two) >= N["client_session"]["params"]["T_client"]):
        qc = [bool(q["cash"] or q["counting"]) for q in Q]
        if np.mean(qc) >= P("solo_cash", "T_solo"):
            best, cur, bs = 0, 0, 0
            for i, c in enumerate(qc):
                cur = cur + 1 if c else 0
                if cur > best: best, bs = cur, i - cur + 1
            add("solo_cash", ts[bs])

    if on("personal_items"):
        pers = [bool(d["personal_item"]) for d in D]
        if longest(pers) / len(ts) >= P("personal_items", "T_pers"):
            best, cur, start, bs = 0, 0, 0, 0
            for i, p in enumerate(pers):
                cur = cur + 1 if p else 0
                if cur == 1: start = i
                if cur > best: best, bs = cur, start
            add("personal_items", t_leave if t_leave is not None else ts[bs])

    if on("door_open") and P("door_open", "T_door_min") < rs < P("door_open", "T_door_max"):
        add("door_open", ts[min(len(ts) - 1, next((i for i, x in enumerate(n) if x >= 1), 0) + 3)])

    if on("cashier_left") and n[-1] == 0 and t_empty is not None:
        if on("laptop_unlocked") and scr >= P("laptop_unlocked", "on_min") and not (sr is not None and sr < P("laptop_unlocked", "T_lock")):
            add("laptop_unlocked", t_leave or t_empty)
        if on("bag_left") and sum(q["cash_bag"] for q in Q) >= P("bag_left", "bag"):
            add("bag_left", t_leave or t_empty)

    if on("safe_open"):
        keys = sorted(sf, key=float); st = [sf[k] for k in keys]
        if sum(x == "A" for x in st[-P("safe_open", "last_k"):]) >= P("safe_open", "min_open"):
            i = len(st) - 1
            while i > 0 and st[i - 1] == "A":
                i -= 1
            add("safe_open", float(keys[i]))
    return ev
