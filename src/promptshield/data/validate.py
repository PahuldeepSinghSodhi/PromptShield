"""pandera schema + span-level checks run before any training step."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from promptshield.types import NER_LABELS


def _spans_valid(row: pd.Series) -> bool:
    text, spans = row["text"], row["spans"]
    prev_end = -1
    for s in spans:
        if not (0 <= s["start"] < s["end"] <= len(text)):
            return False
        if s["start"] < prev_end:  # overlapping or unsorted
            return False
        if s["label"] not in NER_LABELS:
            return False
        if not text[s["start"] : s["end"]].strip():
            return False
        prev_end = s["end"]
    return True


RECORD_SCHEMA = pa.DataFrameSchema(
    {
        "text": pa.Column(str, pa.Check.str_length(min_value=1, max_value=20_000)),
        "spans": pa.Column(object),
        "source": pa.Column(str, pa.Check.isin(["ai4privacy", "synth", "feedback"])),
        "template_id": pa.Column(str, nullable=False),
        "split": pa.Column(str, pa.Check.isin(["train", "val", "test"]), required=False),
    },
    checks=[pa.Check(lambda df: df.apply(_spans_valid, axis=1), error="invalid entity spans")],
    strict=False,
    coerce=False,
)


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    return RECORD_SCHEMA.validate(df, lazy=True)


def check_no_leakage(df: pd.DataFrame) -> None:
    """Every template must live in exactly one split."""
    per_template = df.groupby("template_id")["split"].nunique()
    leaked = per_template[per_template > 1]
    if not leaked.empty:
        raise ValueError(f"{len(leaked)} templates appear in more than one split")


def quality_report(df: pd.DataFrame) -> dict:
    labels = pd.Series([s["label"] for spans in df["spans"] for s in spans])
    return {
        "rows": int(len(df)),
        "by_source": df["source"].value_counts().to_dict(),
        "by_split": df["split"].value_counts().to_dict() if "split" in df else {},
        "rows_without_entities": int((df["spans"].str.len() == 0).sum()),
        "entity_counts": labels.value_counts().to_dict(),
        "text_len_chars": df["text"].str.len().describe().round(1).to_dict(),
        "n_templates": int(df["template_id"].nunique()),
    }
