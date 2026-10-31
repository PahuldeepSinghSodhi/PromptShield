"""High-precision rule detectors with checksum validation.

These complement the ML models for formats where a checksum gives certainty:
Luhn for card numbers, Verhoeff for Aadhaar-style IDs, and fixed formats for
PAN, IFSC and email.
"""

from __future__ import annotations

import re

from promptshield.data.synth import verhoeff_valid
from promptshield.types import Entity

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_AADHAAR = re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")
_PAN = re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b")
_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")


def luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def detect_rules(text: str) -> list[Entity]:
    out: list[Entity] = []

    def add(m: re.Match, label: str, score: float) -> None:
        out.append(Entity(m.start(), m.end(), label, score, "rule", m.group()))

    for m in _EMAIL.finditer(text):
        add(m, "EMAIL", 0.99)
    for m in _AADHAAR.finditer(text):
        if verhoeff_valid(re.sub(r"\D", "", m.group())):
            add(m, "ID_NUMBER", 0.97)
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            add(m, "FINANCIAL", 0.95)
    for m in _PAN.finditer(text):
        add(m, "ID_NUMBER", 0.95)
    for m in _IFSC.finditer(text):
        add(m, "FINANCIAL", 0.93)
    return out
