"""Diagnostics that explain why reserving methods disagree."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from .chainladder import chain_ladder, development_factors
from .triangle import Triangle


def cl_bf_gap(cl_ult: np.ndarray, prior_ult: np.ndarray, cdf: np.ndarray, origins) -> pd.DataFrame:
    """Decompose U_CL - U_BF = (1 - 1/CDF) * (U_CL - U_prior) by accident year.

    The gap is the product of *leverage* (share still to develop) and
    *surprise* (how far chain ladder experience sits from the prior).
    """
    unrep = 1 - 1 / cdf
    surprise = cl_ult - prior_ult
    return pd.DataFrame({
        "pct_unreported": unrep, "cl_ult": cl_ult, "prior_ult": prior_ult,
        "surprise": surprise, "cl_minus_bf": unrep * surprise,
    }, index=pd.Index(origins, name="AY"))


def paid_to_incurred(paid: Triangle, incurred: Triangle) -> pd.DataFrame:
    """Ratio of paid to case incurred at each age.

    Reading down a column holds age fixed: a rising ratio means claims are
    being paid faster *or* case reserves are being set weaker; a falling ratio
    means slower payment or case strengthening.  Either change breaks the
    chain ladder assumption that the pattern is stable across accident years,
    and they push the paid and incurred projections in opposite directions.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = paid.values / incurred.values
    return pd.DataFrame(r, index=list(paid.origins), columns=range(1, paid.n + 1))


def ratio_trend(ratios: pd.DataFrame, ages=(1, 2, 3)) -> pd.DataFrame:
    """Slope of the paid/incurred ratio per accident year at early ages."""
    out = {}
    for a in ages:
        col = ratios[a].dropna()
        if len(col) >= 3:
            res = stats.linregress(col.index.to_numpy(float), col.to_numpy())
            out[a] = {"slope_per_year": res.slope, "p_value": res.pvalue,
                      "first": col.iloc[0], "last": col.iloc[-1]}
    return pd.DataFrame(out).T.rename_axis("age")


def actual_vs_expected_diagonal(tri: Triangle) -> pd.DataFrame:
    """Compare the latest calendar year to what the prior year's factors predicted.

    Fits factors on the triangle as it stood one year earlier, projects one
    step to the current diagonal, and reports actual minus expected emergence.
    A systematic surprise on the newest diagonal is a calendar-year effect
    (inflation shock, claims-handling change, reserve review) that chain
    ladder will extrapolate into every immature accident year.
    """
    n = tri.n
    prior = tri.values.copy()
    for i in range(n):
        prior[i, n - 1 - i] = np.nan
    prior_tri = Triangle(prior[: n - 1, : n - 1], tri.origins[: n - 1])
    f = development_factors(prior_tri)
    rows = []
    for i in range(1, n):  # oldest AY's last factor cannot be estimated a year earlier
        a = n - 2 - i
        if a < 0:
            continue
        expected = tri.values[i, a] * f[a]
        actual = tri.values[i, a + 1]
        rows.append({"AY": tri.origins[i], "age": a + 2, "expected": expected,
                     "actual": actual, "a_minus_e": actual - expected,
                     "a_over_e": actual / expected if expected else np.nan})
    return pd.DataFrame(rows).set_index("AY")


def calendar_year_test(tri: Triangle) -> dict:
    """Mack (1994) test for calendar-year effects in the link ratios.

    Within each development column, link ratios are labelled large or small
    relative to the column median.  Without a calendar-year effect, each
    diagonal should mix large and small; ``Z`` sums min(#large, #small) over
    diagonals and is compared with its null mean and variance.
    """
    lr = tri.link_ratios()
    n = tri.n
    label = np.full(lr.shape, 0)  # +1 large, -1 small, 0 median/unobserved
    for k in range(n - 1):
        rows = n - 1 - k
        col = lr[:rows, k]
        if rows < 2:
            continue
        med = np.median(col)
        label[:rows, k] = np.where(col > med, 1, np.where(col < med, -1, 0))
    z = ez = vz = 0.0
    for d in range(1, n - 1):  # diagonals of link ratios with >= 2 elements
        cells = [(i, d - i) for i in range(d + 1) if d - i < n - 1 and i + (d - i) + 1 <= n - 1]
        lab = np.array([label[i, k] for i, k in cells])
        L, S = int((lab == 1).sum()), int((lab == -1).sum())
        m = L + S
        if m == 0:
            continue
        z += min(L, S)
        c = comb(m - 1, (m - 1) // 2)
        e = m / 2 - c * m / 2**m
        ez += e
        vz += m * (m - 1) / 4 - c * m * (m - 1) / 2**m + e - e**2
    lo, hi = ez - 1.96 * np.sqrt(vz), ez + 1.96 * np.sqrt(vz)
    return {"Z": z, "E[Z]": ez, "ci95": (lo, hi), "calendar_effect": not (lo <= z <= hi)}


def correlation_test(tri: Triangle) -> dict:
    """Mack (1994) test that adjacent development factors are uncorrelated.

    Weighted average of Spearman correlations between neighbouring columns of
    link ratios; flagged if outside the 50% band Mack recommends.
    """
    lr = tri.link_ratios()
    n = tri.n
    num = den = 0.0
    for k in range(1, n - 2):
        rows = n - 1 - k
        if rows < 2:
            continue
        x, y = lr[:rows, k - 1], lr[:rows, k]
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        t = 1 - 6 * np.sum((rx - ry) ** 2) / (rows**3 - rows)
        num += (rows - 1) * t
        den += rows - 1
    T = num / den
    sd = 1 / np.sqrt((n - 2) * (n - 3) / 2)
    band = 0.67 * sd
    return {"T": T, "band50": (-band, band), "correlated": abs(T) > band}


def premium_trend(premium: np.ndarray, origins) -> pd.Series:
    """Year-on-year change in earned premium (book growth / shrinkage)."""
    p = pd.Series(premium, index=list(origins))
    return p.pct_change()


def summarize_triangle(tri: Triangle) -> dict:
    cl = chain_ladder(tri)
    return {"ldf": cl.ldf, "calendar": calendar_year_test(tri), "correlation": correlation_test(tri)}
