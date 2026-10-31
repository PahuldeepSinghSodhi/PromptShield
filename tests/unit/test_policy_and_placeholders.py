from promptshield.policy.engine import Action, PolicyEngine
from promptshield.policy.placeholders import PlaceholderMap
from promptshield.types import Entity

TEXT = "Refund Rahul Mehta at rahul@example.com, then email Rahul Mehta again."


def _ents(score=0.95):
    return [
        Entity(7, 18, "PERSON", score, "ner"),
        Entity(22, 39, "EMAIL", 0.99, "rule"),
        Entity(52, 63, "PERSON", score, "ner"),
    ]


def test_placeholder_round_trip_and_reuse():
    pm = PlaceholderMap()
    red = pm.redact(TEXT, _ents())
    assert red == "Refund [PERSON_1] at [EMAIL_1], then email [PERSON_1] again."
    assert pm.restore(red) == TEXT
    assert pm.summary() == {"PERSON": 1, "EMAIL": 1}


def test_restore_leaves_unknown_tokens():
    pm = PlaceholderMap()
    assert pm.restore("see [PERSON_9]") == "see [PERSON_9]"


def test_policy_redact_default():
    d = PolicyEngine.from_yaml().decide(TEXT, _ents())
    assert d.action == Action.REDACT
    assert "Rahul" not in d.redacted_text


def test_policy_blocks_secrets():
    ents = [Entity(0, 10, "SECRET", 0.99, "secret")]
    d = PolicyEngine.from_yaml().decide("x" * 20, ents)
    assert d.action == Action.BLOCK
    assert d.blocked_types == ["SECRET"]
    assert d.redacted_text == ""


def test_policy_allows_clean_text():
    d = PolicyEngine.from_yaml().decide("Summarise last quarter", [])
    assert d.action == Action.ALLOW


def test_team_override():
    ents = [Entity(0, 5, "USERNAME", 0.95, "ner")]
    engine = PolicyEngine.from_yaml()
    assert engine.decide("alice logs in", ents).action == Action.REDACT
    assert engine.decide("alice logs in", ents, team="engineering").action == Action.ALLOW


def test_low_confidence_dropped_and_review_flag():
    engine = PolicyEngine.from_yaml()
    low = engine.decide(TEXT, [Entity(7, 18, "PERSON", 0.3, "ner")])
    assert low.action == Action.ALLOW and low.entities == []
    d = engine.decide(TEXT, _ents(score=0.6))
    assert d.needs_review
