"""Jury response schema (case §10). Field names must not change."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=100)


class Violation(BaseModel):
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class JobResult(BaseModel):
    job_id: str
    status: Literal["done"] = "done"
    verdict: Literal["violation", "no_violation"]
    violations: list[Violation]

    @model_validator(mode="after")
    def verdict_matches_events(self) -> "JobResult":
        if self.verdict == "violation" and not self.violations:
            raise ValueError("verdict 'violation' needs at least one event")
        if self.verdict == "no_violation" and self.violations:
            raise ValueError("verdict 'no_violation' needs an empty event list")
        return self


class JobStatus(BaseModel):
    job_id: str
    status: Literal["processing", "failed"]
    error: str | None = None


class AnalyzeResponse(BaseModel):
    job_id: str
