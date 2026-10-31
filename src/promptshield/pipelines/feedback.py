"""Turn reviewer feedback into training data without storing real PII.

1. ``mask_with_correction``: the example is masked with placeholders that carry the
   *corrected* labels (false positives are left as plain text, missed entities
   are masked with the label the reviewer gave).
2. ``rehydrate``: before retraining, each placeholder is replaced with a fresh
   fake value of that label, which yields a labelled record with exact spans.
"""

from __future__ import annotations

import random
import re

from faker import Faker

from promptshield.data.synth import _fake_password, _slot_value
from promptshield.policy.placeholders import PlaceholderMap
from promptshield.types import NER_LABELS, Entity

_PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+?)_(\d+)\]")
_FAKE_KIND = {
    "PERSON": "name", "EMAIL": "email", "PHONE": "in_phone", "ADDRESS": "address", "USERNAME": "username",
    "DATE": "date", "ID_NUMBER": "pan", "FINANCIAL": "account",
}


def mask_with_correction(
    example: str, entities: list[Entity], kind: str, value: str | None, corrected_label: str | None
) -> str:
    ents = list(entities)
    if value and value in example:
        vs = example.index(value)
        ve = vs + len(value)
        overlapping = [e for e in ents if e.start < ve and vs < e.end]
        ents = [e for e in ents if e not in overlapping]
        if kind in ("false_negative", "wrong_label") and corrected_label:
            ents.append(Entity(vs, ve, corrected_label, 1.0, "rule", value))
        # false_positive: the value stays as plain text (label O)
    return PlaceholderMap().redact(example, sorted(ents, key=lambda e: e.start))


def rehydrate(masked: str, seed: int = 0) -> dict:
    rng = random.Random(seed)
    fake = Faker("en_IN")
    fake.seed_instance(seed)
    out, spans, cursor, pos = [], [], 0, 0
    for m in _PLACEHOLDER_RE.finditer(masked):
        literal = masked[cursor : m.start()]
        out.append(literal)
        pos += len(literal)
        label = m.group(1)
        if label == "CREDENTIAL":
            v = _fake_password(rng)
        elif label in _FAKE_KIND:
            v = _slot_value(_FAKE_KIND[label], rng, fake)
        else:  # SECRET and unknown labels are not NER targets; keep a neutral token
            v = "REDACTED"
        out.append(v)
        if label in NER_LABELS:
            spans.append({"start": pos, "end": pos + len(v), "label": label})
        pos += len(v)
        cursor = m.end()
    out.append(masked[cursor:])
    return {"text": "".join(out), "spans": spans, "source": "feedback"}
