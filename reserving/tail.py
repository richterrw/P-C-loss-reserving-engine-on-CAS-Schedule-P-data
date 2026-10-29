"""Tail factors beyond the last observed age, and a test of how well curves extrapolate.

Two standard curves are fitted to the development portion (f_j - 1):

  * exponential decay   : f_j - 1 = a exp(-b j)
  * inverse power (Sherman 1984): f_j - 1 = a (j + 1)^-b

``curve_extrapolation_test`` truncates a triangle at age ``k``, fits each
curve to the first ``k`` factors only, and compares the implied development
from age ``k`` to 10 with the factors the triangle actually shows.  That is a
direct test of the step every tail factor relies on.
"""

from __future__ import annotations

import numpy as np

from .chainladder import development_factors
from .triangle import Triangle

CURVES = ("exponential", "inverse_power")


def fit_curve(f: np.ndarray, curve: str, start: int = 0) -> tuple[float, float] | None:
    ages = np.arange(len(f))[start:]
    y = f[start:] - 1.0
    ok = y > 0
    if ok.sum() < 2:
        return None
    x = ages[ok] if curve == "exponential" else np.log(ages[ok] + 1.0)
    slope, icpt = np.polyfit(x, np.log(y[ok]), 1)
    if slope >= 0:
        return None
    return icpt, slope


def curve_factors(params, curve: str, ages: np.ndarray) -> np.ndarray:
    icpt, slope = params
    x = ages if curve == "exponential" else np.log(ages + 1.0)
    return 1.0 + np.exp(icpt + slope * x)


def tail_factor(f: np.ndarray, curve: str = "inverse_power", start: int = 2, horizon: int = 100) -> float:
    """Product of fitted factors from the last observed age out to ``horizon`` more ages."""
    params = fit_curve(f, curve, start)
    if params is None:
        return 1.0
    ages = np.arange(len(f), len(f) + horizon, dtype=float)
    return float(np.prod(curve_factors(params, curve, ages)))


def curve_extrapolation_test(tri: Triangle, k: int = 5, start: int = 1) -> dict:
    """Fit on factors for ages < k, predict the cumulative factor from age k to the end."""
    f = development_factors(tri)
    actual = float(np.prod(f[k:]))
    out = {"actual": actual}
    for c in CURVES:
        params = fit_curve(f[:k], c, start)
        out[c] = float(np.prod(curve_factors(params, c, np.arange(k, len(f), dtype=float)))) if params else 1.0
    return out
