"""Model regression tests: run only when trained artifacts exist (local / retrain flow).

* quality floor on the frozen test split (thresholds from configs/thresholds.yaml)
* invariance: casing / spacing / surrounding text should not change detections
* hard negatives: git SHAs, UUIDs and checksums must not be flagged as secrets
"""

import random
import uuid
from pathlib import Path

import pytest
import yaml

from tests.conftest import fake_aws_key

ROOT = Path(__file__).resolve().parents[2]
HAS_NER = (ROOT / "artifacts" / "ner" / "model.int8.onnx").exists()
HAS_SECRET = (ROOT / "artifacts" / "secret_detector.joblib").exists()
THRESH = yaml.safe_load((ROOT / "configs" / "thresholds.yaml").read_text())["gate"]

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def engine():
    if not (HAS_NER and HAS_SECRET):
        pytest.skip("trained artifacts missing")
    from promptshield.serving.inference import load_engine

    return load_engine("onnx")


@pytest.fixture(scope="module")
def secret_detector():
    if not HAS_SECRET:
        pytest.skip("secret detector missing")
    from promptshield.models.secret_detector import SecretDetector

    return SecretDetector.load()


def test_ner_f1_floor(engine):
    import pandas as pd

    from promptshield.models.benchmark import span_f1

    df = pd.read_parquet(ROOT / "data" / "processed" / "ner.parquet")
    test = df[df.split == "test"].sample(300, random_state=1)
    preds = engine.ner.detect_many(test["text"].tolist())
    f1 = span_f1([list(s) for s in test["spans"]], preds)["f1"]
    assert f1 >= THRESH["min_ner_f1"], f1


@pytest.mark.parametrize("variant", [
    lambda s: s,
    lambda s: "Please help. " + s,
    lambda s: s + " Thanks a lot!",
    lambda s: s.replace(" ", "  "),
])
def test_person_and_email_invariance(engine, variant):
    text = variant("Please email Ananya Iyer at ananya.iyer@example.com about the refund.")
    labels = [e.label for e in engine.detect(text)]
    assert "PERSON" in labels and "EMAIL" in labels


def test_long_document_recall(engine):
    """Chunking regression: a long document must not lose entities at window boundaries."""
    unit = ("Customer Rahul Mehta (rahul.mehta@example.com, +91 98765 43210) says the refund "
            "to account 50100234567812 never arrived. ")
    doc = unit * 60
    ents = [e for e in engine.detect(doc) if e.label in {"PERSON", "EMAIL", "PHONE", "FINANCIAL"}]
    assert len(ents) >= 0.97 * 240
    assert all(doc[e.start : e.end] == e.text for e in ents)  # offsets map back exactly


@pytest.mark.parametrize("seed", range(5))
def test_secret_detected_in_config(secret_detector, seed):
    ents = secret_detector.detect(f"AWS_ACCESS_KEY_ID={fake_aws_key(seed)}")
    assert ents and ents[0].label == "SECRET"


@pytest.mark.parametrize("make", [
    lambda r: f"git checkout {r.getrandbits(160):040x}",
    lambda r: f"request_id={uuid.UUID(int=r.getrandbits(128))}",
    lambda r: f"sha256 checksum {r.getrandbits(256):064x}",
    lambda r: "upgrade pandas to version 2.2.3",
])
def test_hard_negatives_not_flagged(secret_detector, make):
    rng = random.Random(7)
    misses = sum(bool(secret_detector.detect(make(rng))) for _ in range(20))
    assert misses <= 1  # allow at most 5% false positives on look-alikes
