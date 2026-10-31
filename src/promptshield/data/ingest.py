"""Load the public ai4privacy PII dataset into the shared ``text`` / ``spans`` schema."""

from __future__ import annotations

import re

import pandas as pd

from promptshield.data.label_map import LabelMapper

DATASET_ID = "ai4privacy/pii-masking-400k"

# ~30% of the source's English rows have an empty privacy mask, and some of those
# still contain obvious identifiers (16-digit account numbers, emails). Training on
# them as "no PII" teaches the model to miss exactly those values, so unlabeled
# rows that look like they hold an identifier are dropped.
_LIKELY_UNLABELED_PII = re.compile(r"\d[\d -]{7,}\d|[\w.+-]+@[\w-]+\.[\w.]+")


def drop_noisy_negatives(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    noisy = (df["spans"].str.len() == 0) & df["text"].str.contains(_LIKELY_UNLABELED_PII)
    return df[~noisy].reset_index(drop=True), int(noisy.sum())


def load_ai4privacy(
    mapper: LabelMapper, max_rows: int | None = None, seed: int = 42, language: str = "en"
) -> pd.DataFrame:
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset(DATASET_ID)
    ds = concatenate_datasets([ds[s] for s in ds])
    ds = ds.filter(lambda r: r["language"] == language)
    if max_rows is not None and len(ds) > max_rows:
        ds = ds.shuffle(seed=seed).select(range(max_rows))

    rows = []
    for r in ds:
        text = r["source_text"]
        spans = [{"start": m["start"], "end": m["end"], "label": m["label"]} for m in r["privacy_mask"]]
        rows.append({"text": text, "spans": mapper.unify(text, spans), "source": "ai4privacy"})
    return pd.DataFrame(rows)
