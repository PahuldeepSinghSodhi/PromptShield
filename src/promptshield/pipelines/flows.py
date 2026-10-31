"""Prefect flows that orchestrate the data and model pipelines.

    python -m promptshield.pipelines.flows build_dataset
    python -m promptshield.pipelines.flows daily_aggregates [YYYY-MM-DD]
    python -m promptshield.pipelines.flows retrain
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from prefect import flow, get_run_logger, task

from promptshield.paths import ROOT

PY = sys.executable


def _run(module: str, *args: str) -> None:
    subprocess.run([PY, "-m", module, *args], check=True, cwd=ROOT)


# ------------------------------------------------------------------ dataset
@task(retries=2, retry_delay_seconds=30)
def build_datasets_task(max_public: int, n_synth: int, n_secrets: int) -> dict:
    _run("promptshield.data.build", "--max-public", str(max_public), "--n-synth", str(n_synth),
         "--n-secrets", str(n_secrets))
    return json.loads((ROOT / "reports" / "data_quality.json").read_text())


@flow(name="build_dataset")
def build_dataset(max_public: int = 60_000, n_synth: int = 15_000, n_secrets: int = 40_000) -> dict:
    report = build_datasets_task(max_public, n_synth, n_secrets)
    get_run_logger().info("dataset rows=%s entities=%s", report["rows"], report["entity_counts"])
    return report


# ------------------------------------------------------------------ aggregates
@task
def aggregate_day(day: date) -> int:
    from promptshield.db.session import SessionLocal, init_db
    from promptshield.monitoring.aggregates import write_daily

    init_db()
    with SessionLocal() as session:
        return write_daily(session, day)


@flow(name="daily_aggregates")
def daily_aggregates(day: str | None = None) -> int:
    d = date.fromisoformat(day) if day else (datetime.now(UTC) - timedelta(days=1)).date()
    n = aggregate_day(d)
    get_run_logger().info("wrote %d team rows for %s", n, d)
    return n


# ------------------------------------------------------------------ retrain
@task
def merge_feedback() -> int:
    """Append rehydrated feedback rows to the training split of ner.parquet (validated)."""
    from promptshield.data.validate import check_no_leakage, validate_records
    from promptshield.db.session import SessionLocal, init_db
    from promptshield.pipelines.retrain import export_accepted_feedback

    init_db()
    with SessionLocal() as session:
        fb = export_accepted_feedback(session)
    if fb.empty:
        return 0
    path = ROOT / "data" / "processed" / "ner.parquet"
    ner = pd.read_parquet(path)
    ner = ner[ner.source != "feedback"]  # rebuild feedback rows from scratch each run
    merged = pd.concat([ner, fb.assign(split="train")], ignore_index=True)
    validate_records(merged)
    check_no_leakage(merged)
    merged.to_parquet(path, index=False)
    return len(fb)


@task
def train_candidate(epochs: int) -> dict:
    _run("promptshield.models.secret_detector")
    _run("promptshield.models.ner_torch", "--epochs", str(epochs))
    _run("promptshield.models.export_onnx")
    _run("promptshield.models.benchmark", "--n", "300")
    ner = json.loads((ROOT / "reports" / "ner_metrics.json").read_text())
    sec = json.loads((ROOT / "reports" / "secret_detector_metrics.json").read_text())
    bench = json.loads((ROOT / "reports" / "benchmark.json").read_text())
    sel = sec["models"][sec["selected"]]["test"]
    return {
        "ner_f1": ner["test"]["f1"],
        "secret_precision": sel["precision"],
        "secret_recall": sel["recall"],
        "p95_ms": bench["summary"]["onnx_int8"]["p95_ms_batch1"],
    }


@flow(name="retrain")
def retrain(epochs: int = 3) -> dict:
    from promptshield.pipelines.retrain import load_thresholds, regression_gate, write_report

    log = get_run_logger()
    live_metrics_path = ROOT / "reports" / "live_model_metrics.json"
    current = json.loads(live_metrics_path.read_text()) if live_metrics_path.exists() else None

    n_feedback = merge_feedback()
    log.info("merged %d feedback rows", n_feedback)

    # training writes into artifacts/; keep the live models so a failed gate can roll back
    artifacts, backup = ROOT / "artifacts", ROOT / "artifacts.live-backup"
    if backup.exists():
        shutil.rmtree(backup)
    if artifacts.exists():
        shutil.copytree(artifacts, backup)

    candidate = train_candidate(epochs)
    gate = regression_gate(candidate, load_thresholds(), current)
    result = {"feedback_rows": n_feedback, "candidate": candidate, "current": current, "gate": gate}
    if gate["passed"]:
        live_metrics_path.write_text(json.dumps(candidate, indent=2))
        log.info("candidate PROMOTED: %s", candidate)
    else:
        if backup.exists():
            shutil.rmtree(artifacts)
            shutil.move(str(backup), str(artifacts))
        log.warning("candidate REJECTED, live models restored: %s", gate["reasons"])
    result["report"] = str(write_report(result))
    return result


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "daily_aggregates"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    flows = {"build_dataset": build_dataset, "daily_aggregates": daily_aggregates, "retrain": retrain}
    print(flows[name](arg) if (arg and name == "daily_aggregates") else flows[name]())
