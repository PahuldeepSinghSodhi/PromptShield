"""Types shared by detectors, the policy engine and the API."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Unified label set produced by every detector (see configs/label_map.yaml).
ENTITY_LABELS = [
    "PERSON", "EMAIL", "PHONE", "ADDRESS", "USERNAME", "DATE",
    "ID_NUMBER", "FINANCIAL", "CREDENTIAL", "SECRET",
]
NER_LABELS = [lab for lab in ENTITY_LABELS if lab != "SECRET"]  # SECRET comes from sklearn


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    score: float
    source: str  # "ner" | "secret" | "rule"
    text: str = ""

    def to_dict(self, include_text: bool = False) -> dict:
        d = asdict(self)
        if not include_text:
            d.pop("text")
        return d
