"""Client for the local vision-language model server.

Only loopback endpoints are accepted, so a misconfigured URL can never send frames off the machine.
"""

import asyncio
import json
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from app.media import b64

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class VLMError(Exception):
    pass


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(jpeg: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(jpeg)}"}}


class VLM(Protocol):
    async def chat_json(self, parts: list[dict], schema: dict, max_tokens: int) -> dict: ...


def assert_loopback(base_url: str) -> None:
    host = urlparse(base_url).hostname
    if host not in LOOPBACK_HOSTS:
        raise VLMError(f"VLM_BASE_URL must point to this machine, got host {host!r}")


class LocalVLM:
    def __init__(self, base_url: str, model: str, concurrency: int, timeout_sec: float):
        assert_loopback(base_url)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._sem = asyncio.Semaphore(concurrency)
        self._timeout = timeout_sec

    async def chat_json(self, parts: list[dict], schema: dict, max_tokens: int) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": parts}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema},
            },
        }
        last_exc: Exception | None = None
        async with self._sem:
            for _ in range(2):  # one retry for transient errors
                try:
                    async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
                        resp = await client.post(f"{self.base_url}/chat/completions", json=body)
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    return json.loads(content)
                except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                    last_exc = exc
        raise VLMError(f"VLM call failed: {type(last_exc).__name__}")


class MockVLM:
    """TEST ONLY. Returns canned responses; never used for real jury jobs."""

    def __init__(self, responder=None):
        self.responder = responder or (lambda kind, parts: {"events": []})
        self.calls: list[str] = []

    async def chat_json(self, parts: list[dict], schema: dict, max_tokens: int) -> dict:
        kind = "verify" if "status" in schema.get("properties", {}) else "observe"
        self.calls.append(kind)
        return self.responder(kind, parts)


def build_vlm(settings) -> Any:
    if settings.vlm_backend == "mock":
        if not settings.allow_mock:
            raise VLMError("mock VLM requested but ALLOW_MOCK is not set")
        return MockVLM()
    return LocalVLM(settings.vlm_base_url, settings.vlm_model, settings.vlm_concurrency, settings.vlm_timeout_sec)
