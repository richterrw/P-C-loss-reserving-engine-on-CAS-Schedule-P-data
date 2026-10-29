"""Deterministic chain ladder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .triangle import Triangle


def development_factors(tri: Triangle, average: str = "volume") -> np.ndarray:
    """Age-to-age factors ``f_j`` for ``j = 0 .. n-2``.

    ``volume``: sum C[i, j+1] / sum C[i, j] over origins with both observed
    (the Mack / ODP-consistent estimator).  ``simple``: arithmetic mean of the
    individual link ratios.
    """
    v = tri.values
    n = tri.n
    f = np.ones(n - 1)
    for j in range(n - 1):
        rows = slice(0, n - 1 - j)  # origins with age j+1 observed
        num, den = v[rows, j + 1], v[rows, j]
        if average == "volume":
            f[j] = num.sum() / den.sum() if den.sum() != 0 else 1.0
        elif average == "simple":
            ok = den != 0
            f[j] = np.mean(num[ok] / den[ok]) if ok.any() else 1.0
        else:
            raise ValueError(f"unknown average {average!r}")
    return f


def cumulative_factors(f: np.ndarray, tail: float = 1.0) -> np.ndarray:
    """Age-to-ultimate factors; element ``j`` develops age ``j`` to ultimate."""
    return np.cumprod(np.append(f, tail)[::-1])[::-1]


def exponential_tail(f: np.ndarray, fit_from: int = 3, limit: float = 1e-4) -> float:
    """Tail factor from a log-linear fit of (f_j - 1) extrapolated 100 ages out."""
    x = np.arange(len(f))[fit_from:]
    y = f[fit_from:] - 1.0
    ok = y > 0
    if ok.sum() < 2:
        return 1.0
    slope, icpt = np.polyfit(x[ok], np.log(y[ok]), 1)
    if slope >= 0:
        return 1.0
    future = np.exp(icpt + slope * np.arange(len(f), len(f) + 100))
    future = future[future > limit]
    return float(np.prod(1.0 + future))


@dataclass
class ChainLadderResult:
    triangle: Triangle
    ldf: np.ndarray
    cdf: np.ndarray        # age-to-ultimate factor at each origin's latest age
    ultimate: np.ndarray
    tail: float

    @property
    def latest(self) -> np.ndarray:
        return self.triangle.latest

    @property
    def reserve(self) -> np.ndarray:
        """Unreported (to-be-developed) amount on this triangle's basis."""
        return self.ultimate - self.latest

    @property
    def pct_developed(self) -> np.ndarray:
        return 1.0 / self.cdf

    def full_square(self) -> np.ndarray:
        """Triangle completed with chain ladder projections (no tail column)."""
        sq = self.triangle.values.copy()
        n = self.triangle.n
        for i in range(1, n):
            for j in range(n - i, n):
                sq[i, j] = sq[i, j - 1] * self.ldf[j - 1]
        return sq


def chain_ladder(tri: Triangle, average: str = "volume", tail: float | str = 1.0) -> ChainLadderResult:
    f = development_factors(tri, average)
    t = exponential_tail(f) if tail == "exponential" else float(tail)
    cdf_by_age = cumulative_factors(f, t)
    cdf = cdf_by_age[tri.latest_age_index]
    return ChainLadderResult(tri, f, cdf, tri.latest * cdf, t)
