"""Combine NER, secret-detector and rule detections into one non-overlapping list."""

from __future__ import annotations

from promptshield.types import Entity

# When spans overlap: format/checksum-validated rule matches win (a valid PAN is a
# PAN, not an API key), then the more dangerous label, then the higher score.
_PRIORITY = {"SECRET": 3, "CREDENTIAL": 3, "FINANCIAL": 2, "ID_NUMBER": 2}


def fuse(*groups: list[Entity]) -> list[Entity]:
    candidates = [e for g in groups for e in g]
    ranked = sorted(
        candidates,
        key=lambda e: (e.source == "rule", _PRIORITY.get(e.label, 1), e.score, e.end - e.start),
        reverse=True,
    )
    kept: list[Entity] = []
    for e in ranked:
        if all(e.end <= k.start or e.start >= k.end for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start)
