import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_dev import official_score  # noqa: E402


def lab(viol, *times):
    return {"violation": viol, "times": list(times), "events": [{"t": t, "codes": []} for t in times]}


def done(*ts):
    return {"status": "done", "verdict": "violation" if ts else "no_violation",
            "violations": [{"timecode_sec": t} for t in ts]}


def test_perfect_run_scores_one():
    labels = {"a": lab(True, 10.0), "b": lab(False)}
    s = official_score({"a": done(11.0), "b": done()}, labels)
    assert s["score3"] == 1.0 and s["clean_flagged"] == 0


def test_timecode_tolerance_and_extra_events():
    labels = {"a": lab(True, 10.0, 30.0)}
    s = official_score({"a": done(14.0, 31.0, 50.0)}, labels)  # 14 s: only within 5 s; 50 s extra event costs nothing
    assert (s["hit3"], s["hit5"], s["moments"]) == (1, 2, 2)


def test_failed_job_counts_as_wrong():
    labels = {"a": lab(True, 10.0), "b": lab(False)}
    s = official_score({"a": {"status": "failed"}, "b": {"status": "failed"}}, labels)
    assert s["accuracy"] == 0.0 and s["tp"] == 0 and s["hit5"] == 0 and s["clean_flagged"] == 0
