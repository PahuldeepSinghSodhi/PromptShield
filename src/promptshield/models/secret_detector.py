"""scikit-learn secret detector.

Training data comes from running the serving-time candidate extractor over the
synthetic secret texts, so the model sees exactly what it will see in production.

    python -m promptshield.models.secret_detector
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from promptshield.data.synth import generate_secret_challenge
from promptshield.features.secret_features import FEATURE_NAMES, Candidate, extract_candidates, featurize
from promptshield.types import Entity

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "secrets.parquet"
ARTIFACT = ROOT / "artifacts" / "secret_detector.joblib"
METRICS = ROOT / "reports" / "secret_detector_metrics.json"


def candidates_frame(df: pd.DataFrame) -> tuple[list[Candidate], np.ndarray, np.ndarray, pd.Series]:
    """Extract candidates from every text and label them by overlap with the planted span."""
    cands, labels, groups, splits = [], [], [], []
    for row in df.itertuples(index=False):
        for c in extract_candidates(row.text):
            overlaps = c.start < row.end and row.start < c.end
            cands.append(c)
            labels.append(int(row.label == 1 and overlaps))
            groups.append(row.template_id)
            splits.append(row.split)
    return cands, np.asarray(labels), np.asarray(groups), pd.Series(splits)


def pick_threshold(y: np.ndarray, proba: np.ndarray, min_precision: float = 0.98) -> float:
    """Highest-recall threshold whose precision is still >= ``min_precision``.

    Blocking a request by mistake frustrates users, so we favour precision.
    """
    prec, rec, thr = precision_recall_curve(y, proba)
    ok = np.where(prec[:-1] >= min_precision)[0]
    if ok.size == 0:
        return 0.5
    best_recall = rec[:-1][ok].max()
    # among equally good recalls take the highest threshold: fewer false alarms on unseen data
    tied = ok[rec[:-1][ok] >= best_recall - 1e-9]
    return float(thr[tied].max())


def evaluate(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "average_precision": round(float(average_precision_score(y, proba)), 4),
        "threshold": round(threshold, 4),
        "n": int(len(y)),
        "positives": int(y.sum()),
    }


class SecretDetector:
    """Serving wrapper: text in, SECRET entities out."""

    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    @classmethod
    def load(cls, path: str | Path = ARTIFACT) -> SecretDetector:
        bundle = joblib.load(path)
        return cls(bundle["model"], bundle["threshold"])

    def detect(self, text: str) -> list[Entity]:
        cands = extract_candidates(text)
        if not cands:
            return []
        proba = self.model.predict_proba(featurize(cands))[:, 1]
        return [
            Entity(c.start, c.end, "SECRET", float(p), "secret", c.value)
            for c, p in zip(cands, proba, strict=True)
            if p >= self.threshold
        ]


def train() -> dict:
    df = pd.read_parquet(DATA)
    cands, y, groups, split = candidates_frame(df)
    X = featurize(cands)
    tr, va, te = (split == "train").values, (split == "val").values, (split == "test").values

    cv = GroupKFold(n_splits=4)
    searches = {
        "logreg": GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            {"logisticregression__C": [0.1, 1.0, 10.0]},
            scoring="average_precision", cv=cv, n_jobs=-1,
        ),
        "hgb": GridSearchCV(
            HistGradientBoostingClassifier(random_state=42),
            {"learning_rate": [0.05, 0.1], "max_depth": [None, 6], "max_iter": [200]},
            scoring="average_precision", cv=cv, n_jobs=-1,
        ),
    }

    # out-of-distribution challenge set: contexts and formats never seen in training
    ch = generate_secret_challenge(2000).assign(split="challenge")
    ch_cands, y_ch, _, _ = candidates_frame(ch)
    X_ch = featurize(ch_cands)

    results, best_name, best_ap = {}, None, -1.0
    for name, search in searches.items():
        search.fit(X[tr], y[tr], groups=groups[tr])
        val_proba = search.predict_proba(X[va])[:, 1]
        thr = pick_threshold(y[va], val_proba)
        results[name] = {
            "best_params": search.best_params_,
            "val": evaluate(y[va], val_proba, thr),
            "test": evaluate(y[te], search.predict_proba(X[te])[:, 1], thr),
            "challenge_ood": evaluate(y_ch, search.predict_proba(X_ch)[:, 1], thr),
        }
        if results[name]["val"]["average_precision"] > best_ap:
            best_name, best_ap = name, results[name]["val"]["average_precision"]

    best = searches[best_name].best_estimator_
    threshold = results[best_name]["val"]["threshold"]
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "threshold": threshold, "features": FEATURE_NAMES}, ARTIFACT)

    summary = {"selected": best_name, "models": results}
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    print(json.dumps(train(), indent=2, default=str))
