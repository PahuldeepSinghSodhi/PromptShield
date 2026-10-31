"""Text cleaning that keeps entity spans aligned."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+(?:_\d+)?\]")


def normalize_record(text: str, spans: list[dict]) -> tuple[str, list[dict]]:
    """NFKC-normalise text and re-map span offsets character by character.

    NFKC can change string length (e.g. ligatures), so offsets are rebuilt from a
    per-character mapping instead of assuming lengths stay equal.
    """
    out_chars: list[str] = []
    for ch in text:
        norm = unicodedata.normalize("NFKC", ch)
        if norm in ("​", "﻿"):  # zero-width characters
            norm = ""
        out_chars.append(norm)
    # offsets[i] = position of original char i in the normalised text
    offsets, acc = [], 0
    for c in out_chars:
        offsets.append(acc)
        acc += len(c)
    offsets.append(acc)
    new_text = "".join(out_chars)
    new_spans = [
        {"start": offsets[s["start"]], "end": offsets[s["end"]], "label": s["label"]}
        for s in spans
        if offsets[s["end"]] > offsets[s["start"]]
    ]
    return new_text, new_spans


def template_hash(text: str, spans: list[dict]) -> str:
    """Hash of the text with entity values replaced by their labels.

    Records generated from the same template share this hash, which is what the
    leakage-safe split groups on.
    """
    parts, cursor = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        parts.append(text[cursor : s["start"]].lower())
        parts.append(f"<{s['label']}>")
        cursor = s["end"]
    parts.append(text[cursor:].lower())
    skeleton = re.sub(r"\s+", " ", "".join(parts)).strip()
    return hashlib.sha1(skeleton.encode()).hexdigest()[:16]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise, drop empties/duplicates, and attach a template hash where missing."""
    records = [normalize_record(t, s) for t, s in zip(df["text"], df["spans"], strict=True)]
    df = df.assign(text=[r[0] for r in records], spans=[r[1] for r in records])
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset="text")
    hashes = [template_hash(t, s) for t, s in zip(df["text"], df["spans"], strict=True)]
    if "template_id" in df:
        df = df.assign(template_id=df["template_id"].fillna(pd.Series(hashes, index=df.index)))
    else:
        df = df.assign(template_id=hashes)
    return df.reset_index(drop=True)
