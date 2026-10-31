import numpy as np
import pytest

from promptshield.features.secret_features import (
    FEATURE_NAMES,
    extract_candidates,
    featurize,
    shannon_entropy,
)


def test_entropy_bounds():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_key_value_split_gives_value_span():
    text = "export DB_PASSWORD=Xk9mP2vLq7 now"
    cands = extract_candidates(text)
    vals = {c.value: c for c in cands}
    assert "Xk9mP2vLq7" in vals
    c = vals["Xk9mP2vLq7"]
    assert text[c.start : c.end] == "Xk9mP2vLq7"
    assert c.key == "db_password"


def test_quotes_and_trailing_punctuation_are_stripped():
    text = 'token = "abcdef123456".'
    cands = extract_candidates(text)
    assert [c.value for c in cands] == ["abcdef123456"]


def test_short_tokens_are_skipped():
    assert extract_candidates("hi ok yes no") == []


def test_feature_matrix_shape_and_finite():
    cands = extract_candidates("commit 3f2a9c1b7e and api_key=Zx81kLmQ09wErT55")
    X = featurize(cands)
    assert X.shape == (len(cands), len(FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_context_features_differ_for_same_value():
    v = "0123456789abcdef0123456789abcdef"
    def feats(text):
        return featurize([c for c in extract_candidates(text) if c.value == v])[0]

    secret, benign = feats(f"api_key: {v}"), feats(f"md5 checksum {v}")
    i_secret, i_benign = FEATURE_NAMES.index("left_secret_kw"), FEATURE_NAMES.index("left_benign_kw")
    assert secret[i_secret] == 1 and benign[i_benign] == 1
