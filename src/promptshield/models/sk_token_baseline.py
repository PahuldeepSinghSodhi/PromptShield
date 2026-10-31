"""scikit-learn baseline for PII tagging: hand-crafted per-token features + LogisticRegression.

Sets the bar the PyTorch model has to beat, and is evaluated with the same
entity-level seqeval metrics.

    python -m promptshield.models.sk_token_baseline
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "ner.parquet"
ARTIFACT = ROOT / "artifacts" / "sk_token_baseline.joblib"
METRICS = ROOT / "reports" / "sk_token_baseline_metrics.json"

_WORD_RE = re.compile(r"\S+")


def _shape(w: str) -> str:
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def word_tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def word_features(words: list[str], i: int) -> dict:
    w = words[i]
    core = w.strip(".,;:!?()[]\"'")
    f = {
        "bias": 1.0, "lower=" + core.lower(): 1.0, "shape=" + _shape(core): 1.0,
        "pre3=" + core[:3].lower(): 1.0, "suf3=" + core[-3:].lower(): 1.0,
        "is_title": float(core.istitle()), "is_upper": float(core.isupper()),
        "has_digit": float(any(c.isdigit() for c in core)), "has_at": float("@" in core),
        "len_bucket=" + str(min(len(core) // 4, 6)): 1.0,
    }
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(words):
            nb = words[j].strip(".,;:!?()[]\"'")
            f[f"w{off}=" + nb.lower()] = 1.0
            f[f"shape{off}=" + _shape(nb)] = 1.0
        else:
            f[f"w{off}=<pad>"] = 1.0
    return f


def word_bio(text: str, spans: list[dict]) -> tuple[list[str], list[str]]:
    toks = word_tokens(text)
    words = [t[0] for t in toks]
    tags, prev = [], None
    for _, s, e in toks:
        hit = next((sp for sp in spans if sp["start"] < e and s < sp["end"]), None)
        if hit is None:
            tags.append("O")
            prev = None
        else:
            tags.append(("I-" if prev is hit else "B-") + hit["label"])
            prev = hit
    return words, tags


def frame_to_xy(df: pd.DataFrame):
    X, y, seqs = [], [], []
    for text, spans in zip(df["text"], df["spans"], strict=True):
        words, tags = word_bio(text, list(spans))
        X.extend(word_features(words, i) for i in range(len(words)))
        y.extend(tags)
        seqs.append(len(words))
    return X, y, seqs


def _int32_indices(X):
    """DictVectorizer can emit int64 sparse indices, which SGDClassifier rejects."""
    X = X.tocsr()
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int32)
    return X


def _regroup(flat: list[str], lengths: list[int]) -> list[list[str]]:
    out, i = [], 0
    for n in lengths:
        out.append(list(flat[i : i + n]))
        i += n
    return out


def train(max_train: int = 40_000, seed: int = 42) -> dict:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    df = pd.read_parquet(DATA)
    tr = df[df.split == "train"]
    tr = tr.sample(min(max_train, len(tr)), random_state=seed)
    te = df[df.split == "test"]

    Xtr, ytr, _ = frame_to_xy(tr)
    Xte, yte, lte = frame_to_xy(te)
    model = make_pipeline(
        DictVectorizer(sparse=True),
        FunctionTransformer(_int32_indices, accept_sparse=True),
        SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=50, class_weight=None, random_state=seed, n_jobs=-1),
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    y_true, y_pred = _regroup(yte, lte), _regroup(list(pred), lte)
    metrics = {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "per_entity": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "n_train_rows": int(len(tr)),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps({"test": metrics}, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    m = train()
    print({k: m[k] for k in ("precision", "recall", "f1")})
