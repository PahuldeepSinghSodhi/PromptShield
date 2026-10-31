"""PromptShield REST API.

    uvicorn promptshield.serving.app:app --port 8000
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from promptshield import __version__
from promptshield.config import settings
from promptshield.db.models import Event, Feedback, ScanJob
from promptshield.db.session import SessionLocal, get_session, init_db
from promptshield.monitoring import service as monitoring
from promptshield.pipelines.batch_scan import read_table, scan_table
from promptshield.policy.engine import Action, PolicyEngine
from promptshield.serving.inference import load_engine
from promptshield.serving.providers import get_provider
from promptshield.serving.schemas import (
    ChatMessage,
    EntityOut,
    FeedbackOut,
    FeedbackRequest,
    HealthOut,
    JobOut,
    ProxyRequest,
    ProxyResponse,
    RedactResponse,
    ReviewItem,
    ScanRequest,
    ScanResponse,
)
from promptshield.serving.security import authorized

ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------ logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(UTC).isoformat(), "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
log = logging.getLogger("promptshield")
log.handlers = [handler]
log.setLevel(logging.INFO)
log.propagate = False


# ------------------------------------------------------------------ app state
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.engine = load_engine(settings.ner_backend, settings.ner_model_file, settings.ner_threads)
    app.state.policy = PolicyEngine.from_yaml()
    log.info("startup", extra={"extra_fields": {"model_version": app.state.engine.model_version}})
    yield


app = FastAPI(
    title="PromptShield",
    version=__version__,
    description="ML gateway that detects and redacts PII and secrets before prompts reach an LLM.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    if request.url.path.startswith("/v1"):
        log.info("request", extra={"extra_fields": {
            "request_id": request_id, "method": request.method, "path": request.url.path,
            "status": response.status_code, "ms": ms,
        }})
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    log.error("unhandled_error", extra={"extra_fields": {"request_id": rid, "error": type(exc).__name__}})
    return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": rid})


# ------------------------------------------------------------------ helpers
def _check_size(text_value: str) -> None:
    if len(text_value) > settings.max_text_chars:
        raise HTTPException(413, f"text exceeds {settings.max_text_chars} characters")


def _masked_preview(text_value: str, entities, limit: int = 300) -> str:
    from promptshield.policy.placeholders import PlaceholderMap

    return PlaceholderMap().redact(text_value, entities)[:limit]


def _record(session: Session, *, request_id: str, endpoint: str, team: str | None, text_value: str,
            decision, latency_ms: float, model_version: str) -> None:
    counts: dict[str, int] = {}
    for e in decision.entities:
        counts[e.label] = counts.get(e.label, 0) + 1
    session.add(Event(
        request_id=request_id, endpoint=endpoint, team=team,
        text_sha256=hashlib.sha256(text_value.encode()).hexdigest(), text_len=len(text_value),
        action=decision.action.value, entity_counts=counts,
        confidences=[round(e.score, 4) for e in decision.entities], needs_review=decision.needs_review,
        latency_ms=latency_ms, model_version=model_version,
        masked_preview=_masked_preview(text_value, decision.entities) if decision.needs_review else None,
    ))
    session.commit()


async def _scan(request: Request, text_value: str, team: str | None):
    _check_size(text_value)
    engine = request.app.state.engine
    t0 = time.perf_counter()
    entities = await run_in_threadpool(engine.detect, text_value)
    decision = request.app.state.policy.decide(text_value, entities, team)
    return decision, round((time.perf_counter() - t0) * 1000, 2)


def _entities_out(decision) -> list[EntityOut]:
    return [EntityOut(**e.to_dict()) for e in decision.entities]


# ------------------------------------------------------------------ endpoints
@app.get("/v1/health", response_model=HealthOut, tags=["ops"])
def health(request: Request, session: Session = Depends(get_session)):
    try:
        session.execute(text("SELECT 1"))
        db = "ok"
    except Exception:  # noqa: BLE001
        db = "error"
    engine = request.app.state.engine
    return HealthOut(status="ok" if db == "ok" else "degraded", ner_loaded=engine.ner is not None,
                     secret_detector_loaded=engine.secret is not None, db=db)


@app.get("/v1/model-info", tags=["ops"])
def model_info(request: Request, _: str = Depends(authorized)):
    reports = {}
    for name in ("ner_metrics", "secret_detector_metrics", "benchmark"):
        p = ROOT / "reports" / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text())
            reports[name] = {k: data[k] for k in ("test", "calibration", "selected", "summary") if k in data}
    return {"version": __version__, "model_version": request.app.state.engine.model_version,
            "llm_provider": settings.llm_provider, "reports": reports}


@app.post("/v1/scan", response_model=ScanResponse, tags=["detection"])
async def scan(body: ScanRequest, request: Request, session: Session = Depends(get_session),
               _: str = Depends(authorized)):
    decision, ms = await _scan(request, body.text, body.team)
    rid = request.state.request_id
    _record(session, request_id=rid, endpoint="scan", team=body.team, text_value=body.text,
            decision=decision, latency_ms=ms, model_version=request.app.state.engine.model_version)
    return ScanResponse(request_id=rid, entities=_entities_out(decision), action=decision.action.value,
                        needs_review=decision.needs_review, latency_ms=ms,
                        model_version=request.app.state.engine.model_version)


@app.post("/v1/redact", response_model=RedactResponse, tags=["detection"])
async def redact(body: ScanRequest, request: Request, session: Session = Depends(get_session),
                 _: str = Depends(authorized)):
    decision, ms = await _scan(request, body.text, body.team)
    rid = request.state.request_id
    _record(session, request_id=rid, endpoint="redact", team=body.team, text_value=body.text,
            decision=decision, latency_ms=ms, model_version=request.app.state.engine.model_version)
    return RedactResponse(
        request_id=rid, entities=_entities_out(decision), action=decision.action.value,
        needs_review=decision.needs_review, latency_ms=ms, model_version=request.app.state.engine.model_version,
        redacted_text=decision.redacted_text, placeholders=decision.placeholders.summary(),
    )


@app.post("/v1/proxy/chat", response_model=ProxyResponse, tags=["gateway"])
async def proxy_chat(body: ProxyRequest, request: Request, session: Session = Depends(get_session),
                     _: str = Depends(authorized)):
    """Scan every message, apply policy, forward only redacted content, restore the reply."""
    rid = request.state.request_id
    t0 = time.perf_counter()
    policy: PolicyEngine = request.app.state.policy
    engine = request.app.state.engine

    for m in body.messages:
        _check_size(m.content)
    all_entities = await run_in_threadpool(engine.detect_many, [m.content for m in body.messages])

    # one placeholder map across the conversation so [PERSON_1] means the same person everywhere
    from promptshield.policy.placeholders import PlaceholderMap

    shared = PlaceholderMap()
    outgoing, decisions = [], []
    for m, ents in zip(body.messages, all_entities, strict=True):
        d = policy.decide(m.content, ents, body.team)
        decisions.append(d)
        if d.action == Action.REDACT:
            to_redact = [e for e in d.entities if policy.actions_for(body.team).get(e.label) == Action.REDACT]
            outgoing.append(ChatMessage(role=m.role, content=shared.redact(m.content, to_redact)))
        else:
            outgoing.append(ChatMessage(role=m.role, content=m.content))

    blocked = sorted({t for d in decisions for t in d.blocked_types})
    action = Action.BLOCK if blocked else (Action.REDACT if len(shared) else Action.ALLOW)
    counts: dict[str, int] = {}
    for d in decisions:
        for e in d.entities:
            counts[e.label] = counts.get(e.label, 0) + 1

    reply, llm_ms = None, None
    if action != Action.BLOCK:
        t1 = time.perf_counter()
        try:
            raw = await get_provider().chat([m.model_dump() for m in outgoing], body.model)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"LLM provider error: {type(exc).__name__}") from exc
        llm_ms = round((time.perf_counter() - t1) * 1000, 2)
        reply = shared.restore(raw)

    scan_ms = round((time.perf_counter() - t0) * 1000 - (llm_ms or 0), 2)
    joined = "\n".join(m.content for m in body.messages)
    last_entities = _entities_out(decisions[-1])
    merged = decisions[-1]
    merged.action = action
    merged.entities = [e for d in decisions for e in d.entities]
    merged.needs_review = any(d.needs_review for d in decisions)
    _record(session, request_id=rid, endpoint="proxy", team=body.team, text_value=joined, decision=merged,
            latency_ms=scan_ms, model_version=engine.model_version)

    return ProxyResponse(
        request_id=rid, action=action.value, reply=reply, blocked_types=blocked, entities=last_entities,
        sent_to_llm=outgoing if action != Action.BLOCK else None, entity_counts=counts,
        latency_ms=scan_ms, llm_latency_ms=llm_ms,
    )


def _run_scan_job(job_id: str, filename: str, content: bytes, engine) -> None:
    with SessionLocal() as session:
        job = session.get(ScanJob, job_id)
        job.status = "running"
        session.commit()
        try:
            report = scan_table(read_table(filename, content), engine)
            job.status, job.report, job.rows_scanned = "done", report, report["rows_scanned"]
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "failed", str(exc)[:500]
        job.finished_at = datetime.now(UTC)
        session.commit()


@app.post("/v1/scan/batch", response_model=JobOut, status_code=202, tags=["detection"])
async def scan_batch(request: Request, background: BackgroundTasks, file: UploadFile = File(...),
                     session: Session = Depends(get_session), _: str = Depends(authorized)):
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
    if not (file.filename or "").lower().endswith((".csv", ".jsonl", ".ndjson", ".txt")):
        raise HTTPException(415, "unsupported file type; use .csv, .jsonl or .txt")
    job = ScanJob(id=str(uuid.uuid4()), filename=file.filename)
    session.add(job)
    session.commit()
    background.add_task(_run_scan_job, job.id, file.filename, content, request.app.state.engine)
    return JobOut(id=job.id, status=job.status, filename=job.filename, rows_scanned=0)


@app.get("/v1/jobs/{job_id}", response_model=JobOut, tags=["detection"])
def get_job(job_id: str, session: Session = Depends(get_session), _: str = Depends(authorized)):
    job = session.get(ScanJob, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JobOut(id=job.id, status=job.status, filename=job.filename, rows_scanned=job.rows_scanned,
                  report=job.report, error=job.error)


@app.post("/v1/feedback", response_model=FeedbackOut, status_code=201, tags=["feedback"])
async def feedback(body: FeedbackRequest, request: Request, session: Session = Depends(get_session),
                   _: str = Depends(authorized)):
    from promptshield.pipelines.feedback import mask_with_correction

    masked, spans = None, []
    if body.example:
        ents = await run_in_threadpool(request.app.state.engine.detect, body.example)
        masked = mask_with_correction(body.example, ents, body.kind, body.value, body.corrected_label)
        spans = [e.to_dict() for e in ents]
    fb = Feedback(request_id=body.request_id, kind=body.kind, label=body.label,
                  corrected_label=body.corrected_label, masked_example=masked, spans=spans, comment=body.comment)
    session.add(fb)
    session.commit()
    return FeedbackOut(id=fb.id, status=fb.status)


@app.get("/v1/review-queue", response_model=list[ReviewItem], tags=["feedback"])
def review_queue(limit: int = 50, session: Session = Depends(get_session), _: str = Depends(authorized)):
    rows = session.scalars(
        select(Event).where(Event.needs_review.is_(True)).order_by(Event.created_at.desc()).limit(min(limit, 200))
    ).all()
    return [
        ReviewItem(request_id=r.request_id, created_at=r.created_at.isoformat(), team=r.team, action=r.action,
                   entity_counts=r.entity_counts or {},
                   min_confidence=min(r.confidences) if r.confidences else None, masked_preview=r.masked_preview)
        for r in rows
    ]


@app.get("/v1/metrics", tags=["monitoring"])
def get_metrics(hours: int = 24, session: Session = Depends(get_session), _: str = Depends(authorized)):
    return monitoring.metrics(session, hours=min(max(hours, 1), 24 * 30))


@app.get("/v1/drift", tags=["monitoring"])
def get_drift(window: int = 500, session: Session = Depends(get_session), _: str = Depends(authorized)):
    return monitoring.drift(session, window=min(max(window, 50), 10_000))


# ------------------------------------------------------------------ dashboard
_dashboard = ROOT / "dashboard"
if _dashboard.exists():
    app.mount("/", StaticFiles(directory=_dashboard, html=True), name="dashboard")
