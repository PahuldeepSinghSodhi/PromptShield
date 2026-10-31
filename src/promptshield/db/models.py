"""SQL tables. No raw prompt text or entity values are ever stored."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Event(Base):
    """One row per scanned request."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    endpoint: Mapped[str] = mapped_column(String(32))
    team: Mapped[str | None] = mapped_column(String(64), index=True)
    text_sha256: Mapped[str] = mapped_column(String(64))
    text_len: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(16), index=True)
    entity_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    # confidences only (no values) - used for drift monitoring
    confidences: Mapped[list] = mapped_column(JSON, default=list)
    needs_review: Mapped[bool] = mapped_column(default=False, index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(128))
    # masked preview for reviewers, e.g. "Refund for [PERSON_1] to [FINANCIAL_1]"
    masked_preview: Mapped[str | None] = mapped_column(Text)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # false_positive | false_negative | wrong_label
    label: Mapped[str | None] = mapped_column(String(32))
    corrected_label: Mapped[str | None] = mapped_column(String(32))
    # user-supplied example text, PII-masked before storage; used for retraining
    masked_example: Mapped[str | None] = mapped_column(Text)
    spans: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | accepted | rejected
    comment: Mapped[str | None] = mapped_column(Text)


class ScanJob(Base):
    __tablename__ = "scan_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued | running | done | failed
    rows_scanned: Mapped[int] = mapped_column(Integer, default=0)
    report: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)


class DailyAggregate(Base):
    """Written by the Prefect ``daily_aggregates`` flow."""

    __tablename__ = "daily_aggregates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    day: Mapped[str] = mapped_column(String(10), index=True)
    team: Mapped[str] = mapped_column(String(64))
    requests: Mapped[int] = mapped_column(Integer)
    blocked: Mapped[int] = mapped_column(Integer)
    redacted: Mapped[int] = mapped_column(Integer)
    entity_counts: Mapped[dict] = mapped_column(JSON)
    p50_latency_ms: Mapped[float] = mapped_column(Float)
    p95_latency_ms: Mapped[float] = mapped_column(Float)
