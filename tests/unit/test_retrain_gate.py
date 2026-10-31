from promptshield.pipelines.retrain import load_thresholds, regression_gate

GOOD = {"ner_f1": 0.946, "secret_precision": 1.0, "secret_recall": 1.0, "p95_ms": 20.0}


def test_gate_passes_current_model():
    assert regression_gate(GOOD, load_thresholds())["passed"]


def test_gate_rejects_weakened_model():
    weak = {**GOOD, "ner_f1": 0.80}
    result = regression_gate(weak, load_thresholds())
    assert not result["passed"]
    assert any("ner_f1" in r for r in result["reasons"])


def test_gate_rejects_regression_vs_live_model():
    slightly_worse = {**GOOD, "ner_f1": 0.92}  # above the floor but 2.6 points below live
    result = regression_gate(slightly_worse, load_thresholds(), current=GOOD)
    assert not result["passed"]


def test_gate_rejects_slow_model_and_noisy_secrets():
    result = regression_gate({**GOOD, "p95_ms": 90.0, "secret_precision": 0.9}, load_thresholds())
    assert not result["passed"] and len(result["reasons"]) == 2
