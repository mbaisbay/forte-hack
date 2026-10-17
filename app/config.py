"""Runtime settings, read once from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: _env("API_KEY", ""))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    rules_path: Path = field(default_factory=lambda: Path(_env("RULES_PATH", str(ROOT / "config" / "rules.json"))))
    prompts_dir: Path = field(default_factory=lambda: ROOT / "prompts")
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "500")))
    # Shared code for the web UI behind a public URL; empty = UI open (local use). Jury API uses API_KEY.
    ui_access_code: str = field(default_factory=lambda: _env("UI_ACCESS_CODE", ""))

    # Local VLM server (vLLM, OpenAI-compatible). Must be loopback: see vlm.py.
    vlm_base_url: str = field(default_factory=lambda: _env("VLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    vlm_model: str = field(default_factory=lambda: _env("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    vlm_concurrency: int = field(default_factory=lambda: int(_env("VLM_CONCURRENCY", "8")))
    vlm_timeout_sec: float = field(default_factory=lambda: float(_env("VLM_TIMEOUT_SEC", "120")))
    # "mock" is for tests only; the worker refuses it unless ALLOW_MOCK=1.
    vlm_backend: str = field(default_factory=lambda: _env("VLM_BACKEND", "vllm"))
    allow_mock: bool = field(default_factory=lambda: _env("ALLOW_MOCK", "0") == "1")

    # Pipeline knobs (starting points, tune on dev clips).
    scan_fps: float = field(default_factory=lambda: float(_env("SCAN_FPS", "2")))
    verify_fps: float = field(default_factory=lambda: float(_env("VERIFY_FPS", "6")))
    window_sec: float = field(default_factory=lambda: float(_env("WINDOW_SEC", "10")))
    window_overlap_sec: float = field(default_factory=lambda: float(_env("WINDOW_OVERLAP_SEC", "2")))
    verify_pad_sec: float = field(default_factory=lambda: float(_env("VERIFY_PAD_SEC", "3")))
    verify_max_frames: int = field(default_factory=lambda: int(_env("VERIFY_MAX_FRAMES", "36")))
    model_max_side: int = field(default_factory=lambda: int(_env("MODEL_MAX_SIDE", "640")))
    merge_gap_sec: float = field(default_factory=lambda: float(_env("MERGE_GAP_SEC", "3")))
    candidate_min_conf: float = field(default_factory=lambda: float(_env("CANDIDATE_MIN_CONF", "0.3")))
    accept_supported_conf: float = field(default_factory=lambda: float(_env("ACCEPT_SUPPORTED_CONF", "0.5")))
    # Misses are "unacceptable" (FAQ 16), so a confident "unclear" still counts by default.
    accept_unclear_conf: float = field(default_factory=lambda: float(_env("ACCEPT_UNCLEAR_CONF", "0.7")))
    max_verify_calls: int = field(default_factory=lambda: int(_env("MAX_VERIFY_CALLS", "12")))

    # windows = scan + verify only; checklist = whole-clip checklist only; both = union;
    # gated = checklist decides, a scan event survives only near a checklist finding of the same rule (sharper time/text).
    pipeline_mode: str = field(default_factory=lambda: _env("PIPELINE_MODE", "both"))
    checklist_fps: float = field(default_factory=lambda: float(_env("CHECKLIST_FPS", "2")))
    checklist_max_frames: int = field(default_factory=lambda: int(_env("CHECKLIST_MAX_FRAMES", "44")))
    checklist_segment_sec: float = field(default_factory=lambda: float(_env("CHECKLIST_SEGMENT_SEC", "60")))
    checklist_votes: int = field(default_factory=lambda: int(_env("CHECKLIST_VOTES", "2")))
    checklist_agree_sec: float = field(default_factory=lambda: float(_env("CHECKLIST_AGREE_SEC", "4")))
    gate_sec: float = field(default_factory=lambda: float(_env("GATE_SEC", "10")))
    # v2: candidates from checklist votes (union) + scan (visible rules) -> focused re-check on dense frames.
    recheck_max_frames: int = field(default_factory=lambda: int(_env("RECHECK_MAX_FRAMES", "32")))
    recheck_native: bool = field(default_factory=lambda: _env("RECHECK_NATIVE", "1") == "1")
    recheck_min_conf: float = field(default_factory=lambda: float(_env("RECHECK_MIN_CONF", "0.5")))
    recheck_contrast: bool = field(default_factory=lambda: _env("RECHECK_CONTRAST", "1") == "1")
    # Scene-state timeline (door / safe / cash / cashier / customer every ~1.5 s) + persistence rules.
    state_timeline: bool = field(default_factory=lambda: _env("STATE_TIMELINE", "0") == "1")
    state_period_sec: float = field(default_factory=lambda: float(_env("STATE_PERIOD_SEC", "1.5")))
    state_max_frames: int = field(default_factory=lambda: int(_env("STATE_MAX_FRAMES", "40")))
    state_door_min_sec: float = field(default_factory=lambda: float(_env("STATE_DOOR_MIN_SEC", "4")))
    state_away_min_sec: float = field(default_factory=lambda: float(_env("STATE_AWAY_MIN_SEC", "3")))
    # Extra JSON merged into every model request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}.
    vlm_extra_body: str = field(default_factory=lambda: _env("VLM_EXTRA_BODY", ""))

    # Jury cutoff is 300 s end to end; keep margin for queueing and upload.
    job_budget_sec: float = field(default_factory=lambda: float(_env("JOB_BUDGET_SEC", "200")))
    worker_poll_sec: float = field(default_factory=lambda: float(_env("WORKER_POLL_SEC", "0.5")))
    # Jobs processed in parallel; vLLM batches their requests (stress test: 1 job at a time made the 5th of
    # 5 simultaneous long uploads finish after 211 s).
    # Raw model answers per job in results/<job>.calls.jsonl (local diagnostics; never leaves the machine).
    probe_max_tokens: int = field(default_factory=lambda: int(_env("PROBE_MAX_TOKENS", "12000")))
    log_calls: bool = field(default_factory=lambda: _env("LOG_CALLS", "1") == "1")
    worker_threads: int = field(default_factory=lambda: int(_env("WORKER_THREADS", "2")))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.sqlite"


def get_settings() -> Settings:
    return Settings()
