import dataclasses
import hashlib

import pytest
from fastapi.testclient import TestClient

from app import main

CODE = "test-ui-code"


@pytest.fixture
def gated(monkeypatch):
    s = dataclasses.replace(main.settings, ui_access_code=CODE)
    monkeypatch.setattr(main, "settings", s)
    monkeypatch.setattr(main, "_ui_token", hashlib.sha256(f"ui:{CODE}:{s.api_key}".encode()).hexdigest())
    return TestClient(main.app)


def test_ui_open_without_code_configured():
    c = TestClient(main.app)
    assert c.get("/ui/session").json() == {"required": False, "ok": True}
    assert c.get("/ui/jobs/nope").status_code == 404


def test_ui_routes_need_code(gated, synthetic_video):
    assert gated.get("/ui/session").json() == {"required": True, "ok": False}
    assert gated.get("/ui/jobs/nope").status_code == 401
    assert gated.get("/ui/jobs/nope/video").status_code == 401
    with open(synthetic_video, "rb") as f:
        assert gated.post("/ui/analyze", files={"file": ("clip.mov", f, "video/quicktime")}).status_code == 401
    assert gated.get("/").status_code == 200  # the page itself loads and shows the code form


@pytest.mark.parametrize("bad", ["wrong", "", "кириллица"])
def test_wrong_code_rejected(gated, bad):
    assert gated.post("/ui/login", json={"code": bad}).status_code == 401
    assert gated.get("/ui/jobs/nope").status_code == 401


def test_login_sets_cookie_without_exposing_code(gated):
    r = gated.post("/ui/login", json={"code": CODE})
    assert r.status_code == 200
    assert CODE not in r.headers["set-cookie"]
    assert gated.get("/ui/session").json()["ok"] is True
    assert gated.get("/ui/jobs/nope").status_code == 404


def test_link_with_code_logs_in_and_strips_code(gated):
    r = gated.get(f"/?code={CODE}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert gated.get("/ui/jobs/nope").status_code == 404


def test_jury_api_unaffected_and_non_ascii_key_is_401(gated):
    assert gated.get("/api/v1/jobs/nope", headers={"X-API-Key": main.settings.api_key}).status_code == 404
    assert gated.get("/api/v1/jobs/nope", headers={"X-API-Key": "ключ".encode()}).status_code == 401
