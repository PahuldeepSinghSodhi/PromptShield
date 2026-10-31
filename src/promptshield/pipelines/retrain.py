"""Feedback -> training data -> retrain -> regression gate -> promote or reject.

The gate compares a candidate model against thresholds in configs/thresholds.yaml
on the frozen golden test split. Only a passing candidate replaces the served model.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from promptshield.db.models import Feedback

ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS = ROOT / "configs" / "thresholds.yaml"
FEEDBACK_PARQUET = ROOT / "data" / "processed" / "feedback.parquet"
REPORT_DIR = ROOT / "reports" / "retrain"


def load_thresholds() -> dict:
    return yaml.safe_load(THRESHOLDS.read_text())


def export_accepted_feedback(session: Session, copies: int = 5) -> pd.DataFrame:
    """Accepted reviewer corrections become labelled training rows.

    Each masked example is rehydrated ``copies`` times with different fake values,
    so one correction teaches the pattern rather than one specific value.
    """
    from promptshield.data.clean import template_hash
    from promptshield.pipelines.feedback import rehydrate

    rows = session.scalars(select(Feedback).where(Feedback.status == "accepted")).all()
    records = []
    for r in rows:
        if not r.masked_example:
            continue
        for k in range(copies):
            rec = rehydrate(r.masked_example, seed=r.id * 100 + k)
            rec["template_id"] = "fb-" + template_hash(rec["text"], rec["spans"])
            records.append(rec)
    df = pd.DataFrame(records)
    if not df.empty:
        FEEDBACK_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FEEDBACK_PARQUET, index=False)
    return df


def regression_gate(candidate: dict, thresholds: dict, current: dict | None = None) -> dict:
    """Return pass/fail with reasons. ``candidate``/``current`` are metric dicts."""
    reasons = []
    t = thresholds["gate"]
    if candidate["ner_f1"] < t["min_ner_f1"]:
        reasons.append(f"ner_f1 {candidate['ner_f1']} < {t['min_ner_f1']}")
    if candidate["secret_precision"] < t["min_secret_precision"]:
        reasons.append(f"secret_precision {candidate['secret_precision']} < {t['min_secret_precision']}")
    if candidate["secret_recall"] < t["min_secret_recall"]:
        reasons.append(f"secret_recall {candidate['secret_recall']} < {t['min_secret_recall']}")
    if candidate.get("p95_ms") is not None and candidate["p95_ms"] > t["max_p95_ms"]:
        reasons.append(f"p95_ms {candidate['p95_ms']} > {t['max_p95_ms']}")
    if current and candidate["ner_f1"] < current["ner_f1"] - t["max_f1_drop_vs_current"]:
        reasons.append(f"ner_f1 dropped more than {t['max_f1_drop_vs_current']} vs current model")
    return {"passed": not reasons, "reasons": reasons}


def promote(candidate_dir: Path, live_dir: Path) -> Path:
    """Swap in the candidate, keeping the previous model for rollback."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup = live_dir.parent / f"{live_dir.name}.prev-{stamp}"
    if live_dir.exists():
        shutil.move(str(live_dir), str(backup))
    shutil.copytree(candidate_dir, live_dir)
    return backup


def write_report(result: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"retrain-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path
