"""Policy engine: decide allow / redact / block for a set of detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from promptshield.paths import ROOT
from promptshield.policy.placeholders import PlaceholderMap
from promptshield.types import Entity

DEFAULT_POLICY_PATH = ROOT / "configs" / "policies.yaml"


class Action(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass
class Decision:
    action: Action
    redacted_text: str
    entities: list[Entity]
    placeholders: PlaceholderMap
    blocked_types: list[str] = field(default_factory=list)
    needs_review: bool = False


class PolicyEngine:
    def __init__(self, config: dict):
        self.default = config["default"]
        self.teams = config.get("teams", {})

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_POLICY_PATH) -> PolicyEngine:
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def actions_for(self, team: str | None) -> dict[str, Action]:
        actions = dict(self.default["actions"])
        if team and team in self.teams:
            actions.update(self.teams[team].get("actions", {}))
        return {k: Action(v) for k, v in actions.items()}

    def decide(self, text: str, entities: list[Entity], team: str | None = None) -> Decision:
        actions = self.actions_for(team)
        min_conf = self.default["min_confidence"]
        kept = [e for e in entities if e.score >= min_conf]
        needs_review = any(e.score < self.default["review_below"] for e in kept)

        blocked = sorted({e.label for e in kept if actions.get(e.label, Action.REDACT) == Action.BLOCK})
        placeholders = PlaceholderMap()
        if blocked:
            return Decision(Action.BLOCK, "", kept, placeholders, blocked, needs_review)

        to_redact = [e for e in kept if actions.get(e.label, Action.REDACT) == Action.REDACT]
        if not to_redact:
            return Decision(Action.ALLOW, text, kept, placeholders, [], needs_review)

        redacted = placeholders.redact(text, to_redact)
        return Decision(Action.REDACT, redacted, kept, placeholders, [], needs_review)
