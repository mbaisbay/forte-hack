import os

import pytest
from fastapi.testclient import TestClient

from app import main, worker
from app.config import get_settings
from app.rules import RuleRegistry
from app.schemas import JobResult
from app.vlm import MockVLM

KEY = {"X-API-Key": os.environ["API_KEY"]}
client = TestClient(main.app)


def _upload(path, headers=KEY, url="/api/v1/analyze"):
    with open(path, "rb") as f:
        return client.post(url, headers=headers, files={"file": ("clip.mov", f, "video/quicktime")})


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_auth_required_on_both_endpoints(synthetic_video, headers):
    assert _upload(synthetic_video, headers=headers).status_code == 401
    assert client.get("/api/v1/jobs/whatever", headers=headers).status_code == 401


def test_unknown_job_404():
    assert client.get("/api/v1/jobs/nope", headers=KEY).status_code == 404


def test_corrupt_upload_rejected(corrupt_file):
    r = _upload(corrupt_file)
    assert r.status_code == 400


def test_full_flow_processing_then_done(synthetic_video):
    r = _upload(synthetic_video)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}", headers=KEY).json() == {"job_id": job_id, "status": "processing"}

    s = get_settings()
    job = main.store.claim_next()
    worker.process(job, main.store, MockVLM(), RuleRegistry.load(s.rules_path), s)

    body = client.get(f"/api/v1/jobs/{job_id}", headers=KEY).json()
    assert body["status"] == "done" and body["verdict"] == "no_violation" and body["violations"] == []
    JobResult.model_validate(body)
    assert set(body) == {"job_id", "status", "verdict", "violations"}


def test_ui_routes_work_without_key(synthetic_video):
    r = _upload(synthetic_video, headers={}, url="/ui/analyze")
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert client.get(f"/ui/jobs/{job_id}").json()["status"] == "processing"
    assert client.get(f"/ui/jobs/{job_id}/video").status_code == 200
