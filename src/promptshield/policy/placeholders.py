"""Reversible placeholders: ``Rahul Mehta`` -> ``[PERSON_1]`` and back.

The mapping lives only in memory for the duration of one request. It is never
logged or written to the database.
"""

from __future__ import annotations

import re

from promptshield.types import Entity

_PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)_(\d+)\]")


class PlaceholderMap:
    def __init__(self) -> None:
        self._by_value: dict[tuple[str, str], str] = {}
        self._by_token: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._by_token)

    def token_for(self, label: str, value: str) -> str:
        key = (label, value)
        if key not in self._by_value:
            self._counters[label] = self._counters.get(label, 0) + 1
            token = f"[{label}_{self._counters[label]}]"
            self._by_value[key] = token
            self._by_token[token] = value
        return self._by_value[key]

    def redact(self, text: str, entities: list[Entity]) -> str:
        """Replace entity spans with placeholders. The same value always gets the same token."""
        parts, cursor = [], 0
        for e in sorted(entities, key=lambda e: e.start):
            if e.start < cursor:  # overlapping span; the fusion layer should have resolved it
                continue
            parts.append(text[cursor : e.start])
            parts.append(self.token_for(e.label, text[e.start : e.end]))
            cursor = e.end
        parts.append(text[cursor:])
        return "".join(parts)

    def restore(self, text: str) -> str:
        """Put the original values back into an LLM response."""
        return _PLACEHOLDER_RE.sub(lambda m: self._by_token.get(m.group(0), m.group(0)), text)

    def summary(self) -> dict[str, int]:
        return dict(self._counters)
