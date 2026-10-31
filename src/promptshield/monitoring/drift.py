"""Drift indicators implemented with NumPy: PSI and two-sample KS."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def psi_from_counts(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two count (or probability) vectors over the same bins."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = np.clip(e / max(e.sum(), EPS), EPS, None)
    a = np.clip(a / max(a.sum(), EPS), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI for a continuous feature, with bin edges taken from reference quantiles."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, edges)[0]
    cur_counts = np.histogram(current, edges)[0]
    return psi_from_counts(ref_counts, cur_counts)


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    x = np.sort(np.asarray(reference, dtype=float))
    y = np.sort(np.asarray(current, dtype=float))
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return 0.0, 1.0
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / n
    cdf_y = np.searchsorted(y, grid, side="right") / m
    d = float(np.max(np.abs(cdf_x - cdf_y)))
    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)), 0.0, 1.0))
    return d, p


def psi_level(value: float) -> str:
    """Industry rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant shift."""
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"
