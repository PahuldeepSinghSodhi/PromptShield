import numpy as np
import pytest

from promptshield.monitoring.drift import ks_statistic, psi, psi_from_counts, psi_level


def test_psi_zero_for_identical():
    assert psi_from_counts(np.array([10, 20, 30]), np.array([10, 20, 30])) == pytest.approx(0, abs=1e-9)


def test_psi_known_value():
    # hand computed: (0.5-0.25)ln(2) + (0.5-0.75)ln(2/3)
    expected = 0.25 * np.log(2) - 0.25 * np.log(2 / 3)
    assert psi_from_counts(np.array([25, 75]), np.array([50, 50])) == pytest.approx(expected, rel=1e-4)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.1
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.25


def test_ks_matches_scipy():
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 800), rng.normal(0.3, 1, 600)
    d, p = ks_statistic(a, b)
    ref = ks_2samp(a, b)
    assert d == pytest.approx(ref.statistic, abs=1e-9)
    assert p == pytest.approx(ref.pvalue, abs=0.02)


def test_psi_levels():
    assert psi_level(0.05) == "stable"
    assert psi_level(0.2) == "moderate"
    assert psi_level(0.4) == "significant"
