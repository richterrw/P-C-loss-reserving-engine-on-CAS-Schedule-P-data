"""Mack (1993) distribution-free standard error of chain ladder reserves.

Model assumptions, for cumulative losses C[i, k]:
  1. E[C[i,k+1] | C[i,0..k]] = f_k C[i,k]
  2. Var[C[i,k+1] | C[i,0..k]] = sigma_k^2 C[i,k]
  3. accident years are independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chainladder import ChainLadderResult, chain_ladder
from .triangle import Triangle


def mack_sigma2(tri: Triangle, f: np.ndarray) -> np.ndarray:
    v = tri.values
    n = tri.n
    s2 = np.full(n - 1, np.nan)
    for k in range(n - 1):
        m = n - 1 - k  # number of observed link ratios at age k
        if m < 2:
            continue
        c0, c1 = v[:m, k], v[:m, k + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(c0 > 0, c0 * (c1 / c0 - f[k]) ** 2, 0.0)
        s2[k] = terms.sum() / (m - 1)
    # Mack's extrapolation for the last age, where only one ratio exists.
    a = s2[n - 3]
    b = s2[n - 4] if n >= 4 else np.nan
    if np.isfinite(b) and b > 0:
        s2[n - 2] = min(a**2 / b, a, b)
    else:
        s2[n - 2] = a
    return s2


@dataclass
class MackResult:
    cl: ChainLadderResult
    sigma2: np.ndarray
    process_se: np.ndarray
    parameter_se: np.ndarray
    total_se: float
    total_process_se: float
    total_parameter_se: float

    @property
    def reserve(self) -> np.ndarray:
        return self.cl.reserve

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(self.process_se**2 + self.parameter_se**2)

    @property
    def cv(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.se / self.reserve

    def lognormal_params_ultimate(self, latest_total: float) -> tuple[float, float]:
        """(mu, sigma) of a lognormal for total ultimate with mean CL ultimate and Mack SE.

        Fitting the ultimate rather than the reserve keeps the distribution
        valid when the reserve is small or negative (common on incurred data).
        """
        mean, sd = latest_total + self.cl.reserve.sum(), self.total_se
        s2 = np.log1p((sd / mean) ** 2)
        return np.log(mean) - s2 / 2, np.sqrt(s2)


def mack(tri: Triangle) -> MackResult:
    """Mack standard errors for a volume-weighted chain ladder with no tail."""
    cl = chain_ladder(tri, "volume", tail=1.0)
    f, n, v = cl.ldf, tri.n, tri.values
    s2 = mack_sigma2(tri, f)
    sq = cl.full_square()
    ult = sq[:, -1]
    col_sum = np.array([v[: n - 1 - k, k].sum() for k in range(n - 1)])  # S_k
    w = s2 / f**2

    proc = np.zeros(n)
    param = np.zeros(n)
    for i in range(1, n):
        ks = np.arange(n - 1 - i, n - 1)
        proc[i] = ult[i] ** 2 * np.sum(w[ks] / sq[i, ks])
        param[i] = ult[i] ** 2 * np.sum(w[ks] / col_sum[ks])

    # Total: parameter errors are correlated across origins via shared f_k.
    cross = 0.0
    for i in range(1, n):
        ks = np.arange(n - 1 - i, n - 1)
        cross += ult[i] * ult[i + 1 :].sum() * np.sum(2 * w[ks] / col_sum[ks])
    tot_proc = proc.sum()
    tot_param = param.sum() + cross
    return MackResult(
        cl, s2, np.sqrt(proc), np.sqrt(param),
        float(np.sqrt(tot_proc + tot_param)), float(np.sqrt(tot_proc)), float(np.sqrt(tot_param)),
    )
