"""Pydantic request/response models (these also generate the OpenAPI docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from promptshield.types import ENTITY_LABELS


class EntityOut(BaseModel):
    start: int
    end: int
    label: str
    score: float
    source: Literal["ner", "secret", "rule"]


class ScanRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Refund Rahul Mehta at rahul@example.com"])
    team: str | None = Field(None, examples=["finance"])


class ScanResponse(BaseModel):
    request_id: str
    entities: list[EntityOut]
    action: Literal["allow", "redact", "block"]
    needs_review: bool
    latency_ms: float
    model_version: str


class RedactResponse(ScanResponse):
    redacted_text: str
    placeholders: dict[str, int] = Field(description="Placeholder counts per label (values are never returned)")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ProxyRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    team: str | None = None
    model: str | None = None


class ProxyResponse(BaseModel):
    request_id: str
    action: Literal["allow", "redact", "block"]
    reply: str | None = Field(None, description="LLM reply with placeholders restored")
    blocked_types: list[str] = []
    entities: list[EntityOut] = Field([], description="Detections in the last message (offsets only)")
    sent_to_llm: list[ChatMessage] | None = Field(None, description="Exactly what left the gateway")
    entity_counts: dict[str, int]
    latency_ms: float
    llm_latency_ms: float | None = None


class FeedbackRequest(BaseModel):
    request_id: str
    kind: Literal["false_positive", "false_negative", "wrong_label"]
    label: str | None = Field(None, description="Label PromptShield predicted")
    corrected_label: str | None = Field(None, description=f"One of {ENTITY_LABELS} or null for 'not sensitive'")
    example: str | None = Field(None, max_length=2000, description="Optional example; PII is masked before storage")
    value: str | None = Field(
        None, max_length=200,
        description="Exact text that was wrongly flagged or missed. Used to build the masked example; never stored.",
    )
    comment: str | None = Field(None, max_length=1000)


class FeedbackOut(BaseModel):
    id: int
    status: str


class ReviewItem(BaseModel):
    request_id: str
    created_at: str
    team: str | None
    action: str
    entity_counts: dict[str, int]
    min_confidence: float | None
    masked_preview: str | None


class JobOut(BaseModel):
    id: str
    status: str
    filename: str
    rows_scanned: int
    report: dict | None = None
    error: str | None = None


class HealthOut(BaseModel):
    status: str
    ner_loaded: bool
    secret_detector_loaded: bool
    db: str
