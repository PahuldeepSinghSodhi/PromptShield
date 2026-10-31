"""Metrics and drift computed from the events table."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from promptshield.db.models import Event
from promptshield.monitoring.drift import ks_statistic, psi, psi_from_counts, psi_level
from promptshield.paths import ROOT
from promptshield.types import ENTITY_LABELS

REFERENCE_PATH = ROOT / "artifacts" / "drift_reference.json"


def events_frame(session: Session, since: datetime | None = None, limit: int = 5000) -> pd.DataFrame:
    q = select(Event).order_by(Event.created_at.desc()).limit(limit)
    if since is not None:
        q = q.where(Event.created_at >= since)
    rows = session.scalars(q).all()
    return pd.DataFrame(
        [
            {
                "created_at": r.created_at, "team": r.team or "unknown", "action": r.action,
                "entity_counts": r.entity_counts or {}, "confidences": r.confidences or [],
                "latency_ms": r.latency_ms, "text_len": r.text_len, "endpoint": r.endpoint,
            }
            for r in rows
        ]
    )


def metrics(session: Session, hours: int = 24) -> dict:
    df = events_frame(session, since=datetime.now(UTC) - timedelta(hours=hours))
    if df.empty:
        return {"window_hours": hours, "requests": 0}
    entity_totals: Counter[str] = Counter()
    for c in df["entity_counts"]:
        entity_totals.update(c)
    df["hour"] = pd.to_datetime(df["created_at"], utc=True).dt.floor("h")
    by_hour = df.groupby(["hour", "action"]).size().unstack(fill_value=0)
    return {
        "window_hours": hours,
        "requests": int(len(df)),
        "actions": df["action"].value_counts().to_dict(),
        "entities": dict(entity_totals.most_common()),
        "by_team": df.groupby("team")["action"].value_counts().unstack(fill_value=0).to_dict(orient="index"),
        "latency_ms": {
            "p50": round(float(df["latency_ms"].quantile(0.5)), 2),
            "p95": round(float(df["latency_ms"].quantile(0.95)), 2),
            "p99": round(float(df["latency_ms"].quantile(0.99)), 2),
        },
        "timeline": [
            {"hour": h.isoformat(), **{k: int(v) for k, v in row.items()}} for h, row in by_hour.iterrows()
        ],
    }


def build_reference(entity_lists: list[list], text_lens: list[int]) -> dict:
    """Reference distributions from a held-out set run through the engine (see scripts)."""
    label_counts = Counter(e.label for ents in entity_lists for e in ents)
    return {
        "label_counts": {lab: label_counts.get(lab, 0) for lab in ENTITY_LABELS},
        "confidences": [round(e.score, 4) for ents in entity_lists for e in ents],
        "text_len": list(map(int, text_lens)),
        "entities_per_request": [len(ents) for ents in entity_lists],
    }


def save_reference(ref: dict, path: Path = REFERENCE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref))


def drift(session: Session, window: int = 500, path: Path = REFERENCE_PATH) -> dict:
    if not path.exists():
        return {"status": "no_reference", "detail": "run scripts/build_drift_reference.py"}
    ref = json.loads(path.read_text())
    df = events_frame(session, limit=window)
    if len(df) < 50:
        return {"status": "insufficient_data", "events": int(len(df)), "min_events": 50}

    cur_counts: Counter[str] = Counter()
    for c in df["entity_counts"]:
        cur_counts.update(c)
    cur_conf = np.asarray([x for xs in df["confidences"] for x in xs], dtype=float)
    cur_per_req = df["entity_counts"].apply(lambda c: sum(c.values())).to_numpy()

    checks = {}
    checks["entity_mix"] = {
        "psi": round(psi_from_counts(
            np.array([ref["label_counts"][lab] for lab in ENTITY_LABELS]),
            np.array([cur_counts.get(lab, 0) for lab in ENTITY_LABELS]),
        ), 4)
    }
    for name, ref_vals, cur_vals in [
        ("text_length", ref["text_len"], df["text_len"].to_numpy()),
        ("confidence", ref["confidences"], cur_conf),
        ("entities_per_request", ref["entities_per_request"], cur_per_req),
    ]:
        if len(cur_vals) == 0 or len(ref_vals) == 0:
            continue
        d, p = ks_statistic(np.asarray(ref_vals), cur_vals)
        checks[name] = {"psi": round(psi(np.asarray(ref_vals), cur_vals), 4), "ks": round(d, 4), "ks_p": round(p, 4)}
    for c in checks.values():
        c["level"] = psi_level(c["psi"])
    worst = max(checks.values(), key=lambda c: c["psi"])["level"]
    return {"status": "ok", "events": int(len(df)), "overall": worst, "checks": checks}
