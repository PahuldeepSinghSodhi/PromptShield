"""Scan an uploaded file (CSV / JSONL / TXT) and report where sensitive data lives.

The report contains counts and row/column locations only, never the values.
"""

from __future__ import annotations

import io
from collections import Counter, defaultdict

import pandas as pd

from promptshield.serving.inference import DetectionEngine

TEXT_BATCH = 64


def read_table(filename: str, content: bytes) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    if name.endswith((".jsonl", ".ndjson")):
        return pd.read_json(io.BytesIO(content), lines=True, dtype=str).fillna("")
    if name.endswith(".txt"):
        lines = content.decode("utf-8", errors="replace").splitlines()
        return pd.DataFrame({"line": lines})
    raise ValueError("unsupported file type; use .csv, .jsonl or .txt")


def scan_table(df: pd.DataFrame, engine: DetectionEngine, max_rows: int = 20_000) -> dict:
    df = df.head(max_rows).astype(str)
    per_column: dict[str, Counter] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    rows_with_hits: set[int] = set()
    sample_locations: list[dict] = []

    cells = [(i, col, v) for col in df.columns for i, v in df[col].items() if v.strip()]
    for b in range(0, len(cells), TEXT_BATCH):
        chunk = cells[b : b + TEXT_BATCH]
        results = engine.detect_many([v for _, _, v in chunk])
        for (i, col, _), ents in zip(chunk, results, strict=True):
            if not ents:
                continue
            rows_with_hits.add(int(i))
            for e in ents:
                per_column[col][e.label] += 1
                totals[e.label] += 1
            if len(sample_locations) < 50:
                sample_locations.append({"row": int(i), "column": col, "labels": sorted({e.label for e in ents})})

    risky_columns = sorted(per_column, key=lambda c: -sum(per_column[c].values()))
    return {
        "rows_scanned": int(len(df)),
        "columns": list(df.columns),
        "rows_with_sensitive_data": len(rows_with_hits),
        "pct_rows_with_sensitive_data": round(100 * len(rows_with_hits) / max(1, len(df)), 2),
        "entity_totals": dict(totals.most_common()),
        "per_column": {c: dict(per_column[c]) for c in risky_columns},
        "sample_locations": sample_locations,
        "contains_secrets": totals.get("SECRET", 0) + totals.get("CREDENTIAL", 0) > 0,
    }
