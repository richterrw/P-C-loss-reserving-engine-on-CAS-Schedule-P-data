"""Munich Chain Ladder (Quarg & Mack, 2004).

Separate paid and incurred chain ladders ignore each other, so their
ultimates drift apart (section 2 of the README).  MCL uses the observed
correlation between a year's paid/incurred ratio and its next development
factor: a year whose paid is unusually low relative to incurred tends to have
high paid development next period, and vice versa.

For residuals standardised with Mack's variance assumptions,

    lambda_P = slope of Res(P_{j+1}/P_j) on Res(I_j/P_j)   (through the origin)
    lambda_I = slope of Res(I_{j+1}/I_j) on Res(P_j/I_j)

and future cells are projected jointly:

    P_{j+1} = P_j [ f^P_j + lambda_P sigma^P_j / rho^P_j (I_j/P_j - q^-1_j) ]
    I_{j+1} = I_j [ f^I_j + lambda_I sigma^I_j / rho^I_j (P_j/I_j - q_j) ]

with q_j = sum P / sum I at age j and rho the Mack-style volatility of the
ratios.  lambda = 0 reproduces the two separate chain ladders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chainladder import development_factors
from .mack import mack_sigma2
from .triangle import Triangle


def _ratio_rho2(num: Triangle, den: Triangle, weight: Triangle) -> tuple[np.ndarray, np.ndarray]:
    """Volume-weighted ratio sum(num)/sum(den) per age and its Mack-style variance."""
    n = num.n
    q = np.empty(n)
    rho2 = np.full(n, np.nan)
    for j in range(n):
        rows = n - j
        q[j] = num.values[:rows, j].sum() / den.values[:rows, j].sum()
        if rows >= 2:
            x = num.values[:rows, j] / den.values[:rows, j]
            w = weight.values[:rows, j]
            rho2[j] = np.sum(w * (x - q[j]) ** 2) / (rows - 1)
    # The oldest age has a single observation; extrapolate log-linearly.
    ok = np.isfinite(rho2) & (rho2 > 0)
    ages = np.arange(n)
    b = np.polyfit(ages[ok], np.log(np.sqrt(rho2[ok])), 1)
    rho2[n - 1] = np.exp(np.polyval(b, n - 1)) ** 2
    return q, rho2


@dataclass
class MunichResult:
    paid_square: np.ndarray
    incurred_square: np.ndarray
    lambda_paid: float
    lambda_incurred: float

    @property
    def paid_ultimate(self) -> np.ndarray:
        return self.paid_square[:, -1]

    @property
    def incurred_ultimate(self) -> np.ndarray:
        return self.incurred_square[:, -1]


def munich_chain_ladder(paid: Triangle, incurred: Triangle, sigma_extrapolation: str = "log-linear") -> MunichResult:
    n = paid.n
    P, I = paid.values, incurred.values
    fP, fI = development_factors(paid), development_factors(incurred)
    sP = np.sqrt(mack_sigma2(paid, fP, sigma_extrapolation))
    sI = np.sqrt(mack_sigma2(incurred, fI, sigma_extrapolation))
    inv_ratio = Triangle(I / P, paid.origins)       # Q^-1 = I/P
    ratio = Triangle(P / I, paid.origins)           # Q = P/I
    q, rhoI2 = _ratio_rho2(paid, incurred, incurred)   # Q, weighted by I
    qinv, rhoP2 = _ratio_rho2(incurred, paid, paid)    # Q^-1, weighted by P
    rhoP, rhoI = np.sqrt(rhoP2), np.sqrt(rhoI2)
    # Correction slopes per age; zero (plain chain ladder) where a volatility is degenerate.
    with np.errstate(divide="ignore", invalid="ignore"):
        kP = sP / rhoP[: n - 1]
        kI = sI / rhoI[: n - 1]
    kP = np.where(np.isfinite(kP), kP, 0.0)
    kI = np.where(np.isfinite(kI), kI, 0.0)

    # Residual pairs for the lambda regressions.  Ages with a single link ratio
    # are skipped: that ratio *is* the factor, so its residual is zero by construction.
    xs_p, ys_p, xs_i, ys_i = [], [], [], []
    for j in range(n - 1):
        rows = n - 1 - j
        if rows < 2 or not (kP[j] > 0 and kI[j] > 0):
            continue
        p0, p1, i0, i1 = P[:rows, j], P[:rows, j + 1], I[:rows, j], I[:rows, j + 1]
        ys_p.append((p1 / p0 - fP[j]) / sP[j] * np.sqrt(p0))
        xs_p.append((inv_ratio.values[:rows, j] - qinv[j]) / rhoP[j] * np.sqrt(p0))
        ys_i.append((i1 / i0 - fI[j]) / sI[j] * np.sqrt(i0))
        xs_i.append((ratio.values[:rows, j] - q[j]) / rhoI[j] * np.sqrt(i0))
    if xs_p:
        xp, yp, xi, yi = (np.concatenate(v) for v in (xs_p, ys_p, xs_i, ys_i))
        lam_p = float(np.sum(xp * yp) / np.sum(xp**2))
        lam_i = float(np.sum(xi * yi) / np.sum(xi**2))
    else:
        lam_p = lam_i = 0.0

    Psq, Isq = P.copy(), I.copy()
    for i in range(1, n):
        for j in range(n - 1 - i, n - 1):
            p, c = Psq[i, j], Isq[i, j]
            if p <= 0 or c <= 0:  # ratios undefined: fall back to plain chain ladder
                Psq[i, j + 1], Isq[i, j + 1] = p * fP[j], c * fI[j]
                continue
            Psq[i, j + 1] = p * (fP[j] + lam_p * kP[j] * (c / p - qinv[j]))
            Isq[i, j + 1] = c * (fI[j] + lam_i * kI[j] * (p / c - q[j]))
    return MunichResult(Psq, Isq, lam_p, lam_i)
