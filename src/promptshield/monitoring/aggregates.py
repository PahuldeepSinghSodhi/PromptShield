"""Daily roll-ups of the events table (pandas), written to ``daily_aggregates``."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from promptshield.db.models import DailyAggregate, Event


def compute_daily(session: Session, day: date) -> pd.DataFrame:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    end = start + timedelta(days=1)
    rows = session.execute(
        select(Event.team, Event.action, Event.entity_counts, Event.latency_ms)
        .where(Event.created_at >= start, Event.created_at < end)
    ).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["team", "action", "entity_counts", "latency_ms"])
    df["team"] = df["team"].fillna("unknown")

    out = []
    for team, g in df.groupby("team"):
        counts: Counter[str] = Counter()
        for c in g["entity_counts"]:
            counts.update(c or {})
        out.append({
            "day": day.isoformat(), "team": team, "requests": len(g),
            "blocked": int((g["action"] == "block").sum()), "redacted": int((g["action"] == "redact").sum()),
            "entity_counts": dict(counts),
            "p50_latency_ms": round(float(g["latency_ms"].quantile(0.5)), 2),
            "p95_latency_ms": round(float(g["latency_ms"].quantile(0.95)), 2),
        })
    return pd.DataFrame(out)


def write_daily(session: Session, day: date) -> int:
    """Idempotent: re-running a day replaces its rows."""
    df = compute_daily(session, day)
    session.execute(delete(DailyAggregate).where(DailyAggregate.day == day.isoformat()))
    for rec in df.to_dict(orient="records"):
        session.add(DailyAggregate(**rec))
    session.commit()
    return len(df)
