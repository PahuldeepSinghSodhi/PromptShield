"""Build the NER and secret-detector datasets end to end.

    python -m promptshield.data.build --max-public 60000 --n-synth 15000 --n-secrets 40000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from promptshield.data.clean import clean_frame
from promptshield.data.ingest import load_ai4privacy
from promptshield.data.label_map import LabelMapper
from promptshield.data.split import grouped_split
from promptshield.data.synth import generate_pii_records, generate_secret_texts
from promptshield.data.validate import check_no_leakage, quality_report, validate_records

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def build_ner(max_public: int, n_synth: int, seed: int) -> tuple[pd.DataFrame, dict]:
    mapper = LabelMapper()
    public = load_ai4privacy(mapper, max_rows=max_public, seed=seed)
    synth = generate_pii_records(n_synth, seed=seed)
    df = clean_frame(pd.concat([public, synth], ignore_index=True))
    df = grouped_split(df, seed=seed)
    validate_records(df)
    check_no_leakage(df)
    report = quality_report(df)
    report["unmapped_source_labels"] = dict(mapper.unmapped.most_common())
    return df, report


def build_secrets(n: int, seed: int) -> pd.DataFrame:
    df = generate_secret_texts(n, seed=seed)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df = grouped_split(df, seed=seed)
    check_no_leakage(df)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-public", type=int, default=60_000)
    ap.add_argument("--n-synth", type=int, default=15_000)
    ap.add_argument("--n-secrets", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    ner, report = build_ner(args.max_public, args.n_synth, args.seed)
    ner.to_parquet(OUT / "ner.parquet", index=False)
    secrets = build_secrets(args.n_secrets, args.seed)
    secrets.to_parquet(OUT / "secrets.parquet", index=False)

    report["secrets"] = {
        "rows": int(len(secrets)),
        "positive_rate": round(float(secrets["label"].mean()), 3),
        "by_split": secrets["split"].value_counts().to_dict(),
    }
    (REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
