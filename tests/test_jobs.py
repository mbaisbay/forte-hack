from app.jobs import JobStore


def test_claim_finish_and_stale_write(tmp_path):
    store = JobStore(tmp_path / "j.sqlite")
    store.create("a", tmp_path / "a.mov")
    job = store.claim_next()
    assert job.state == "running" and job.attempt == 1
    assert store.claim_next() is None
    stale = job
    assert store.recover_interrupted() == 1          # simulate a worker restart
    fresh = store.claim_next()
    assert fresh.attempt == 2
    assert store.finish(stale, tmp_path / "r.json") is False   # old claim cannot overwrite
    assert store.finish(fresh, tmp_path / "r.json") is True
    assert store.get("a").state == "done"


def test_repeatedly_interrupted_job_fails(tmp_path):
    store = JobStore(tmp_path / "j.sqlite")
    store.create("b", tmp_path / "b.mov")
    store.claim_next(); store.recover_interrupted()
    store.claim_next(); store.recover_interrupted()
    assert store.get("b").state == "failed"
