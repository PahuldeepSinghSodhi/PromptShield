"""Unify entity labels across data sources."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import yaml

LABEL_MAP_PATH = Path(__file__).resolve().parents[3] / "configs" / "label_map.yaml"


class LabelMapper:
    def __init__(self, path: str | Path = LABEL_MAP_PATH):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self._map = {src.upper(): tgt for tgt, srcs in cfg["unified"].items() for src in srcs}
        self._gap = re.compile(cfg["merge_adjacent_gap"])
        self.unmapped: Counter[str] = Counter()

    def map_label(self, label: str) -> str | None:
        tgt = self._map.get(label.upper())
        if tgt is None:
            self.unmapped[label] += 1
        return tgt

    def unify(self, text: str, spans: list[dict]) -> list[dict]:
        """Map labels, drop unmapped ones, and merge adjacent same-label spans."""
        mapped = []
        for s in sorted(spans, key=lambda s: s["start"]):
            tgt = self.map_label(s["label"])
            if tgt is not None:
                mapped.append({"start": int(s["start"]), "end": int(s["end"]), "label": tgt})
        merged: list[dict] = []
        for s in mapped:
            if merged:
                prev = merged[-1]
                if s["start"] < prev["end"]:  # overlap: keep the earlier span
                    continue
                if s["label"] == prev["label"] and self._gap.match(text[prev["end"] : s["start"]]):
                    prev["end"] = s["end"]
                    continue
            merged.append(dict(s))
        return merged
