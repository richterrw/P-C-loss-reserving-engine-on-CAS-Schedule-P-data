"""Over-dispersed Poisson (ODP) bootstrap of chain ladder reserves.

Follows England & Verrall (2002) and Shapland (2016):

1. Fit the volume-weighted chain ladder and back-cast fitted cumulative
   values from the latest diagonal; difference them into fitted incrementals m.
2. Unscaled Pearson residuals r = (q - m) / sqrt(|m|); scale parameter
   phi = sum r^2 / (N - p) with N observed cells and p = 2n - 1 parameters.
   Residuals get the degrees-of-freedom adjustment sqrt(N / (N - p)).
3. For each simulation, resample residuals into every observed cell,
   rebuild a pseudo triangle q* = m + r* sqrt(|m|), refit chain ladder factors
   (parameter risk) and project future incremental means mu*.
4. Add process risk by drawing each future incremental from a gamma with
   mean mu* and variance phi * mu* (sign-flipped when mu* < 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chainladder import chain_ladder, cumulative_factors
from .triangle import Triangle, upper_mask


@dataclass
class BootstrapResult:
    reserves: np.ndarray       # (n_sims, n_origins) simulated unpaid by origin
    phi: float
    residuals: np.ndarray      # adjusted Pearson residuals, NaN where unobserved
    deterministic: np.ndarray  # chain ladder reserve by origin

    @property
    def total(self) -> np.ndarray:
        return self.reserves.sum(axis=1)

    @property
    def mean(self) -> np.ndarray:
        return self.reserves.mean(axis=0)

    @property
    def se(self) -> np.ndarray:
        return self.reserves.std(axis=0, ddof=1)

    @property
    def total_se(self) -> float:
        return float(self.total.std(ddof=1))

    def percentile(self, q) -> np.ndarray:
        return np.percentile(self.total, q)

    def rank_of(self, actual: float) -> float:
        """Share of simulated totals at or below ``actual`` (predictive percentile)."""
        return float((self.total <= actual).mean())


def fitted_incrementals(tri: Triangle) -> np.ndarray:
    """ODP/chain-ladder fitted incremental values on the observed cells."""
    cl = chain_ladder(tri, "volume", tail=1.0)
    cdf_age = cumulative_factors(cl.ldf, 1.0)
    n = tri.n
    m_cum = np.full((n, n), np.nan)
    for i in range(n):
        a = n - 1 - i
        m_cum[i, : a + 1] = tri.latest[i] * cdf_age[a] / cdf_age[: a + 1]
    return np.diff(m_cum, axis=1, prepend=0.0)


def odp_bootstrap(tri: Triangle, n_sims: int = 5000, seed: int | None = 12345, process: bool = True) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    n = tri.n
    obs = upper_mask(n)
    q = tri.incremental
    m = fitted_incrementals(tri)
    root = np.sqrt(np.abs(m))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(obs & (root > 0), (q - m) / root, np.nan)
    r[~obs] = np.nan

    n_obs = int(obs.sum())
    n_par = 2 * n - 1
    dof = n_obs - n_par
    phi = float(np.nansum(r**2) / dof)
    r_adj = r * np.sqrt(n_obs / dof)

    # Corner cells are fitted exactly (residual identically zero); keep them out of the pool.
    pool_mask = obs.copy()
    pool_mask[0, n - 1] = pool_mask[n - 1, 0] = False
    pool = r_adj[pool_mask & np.isfinite(r_adj)]

    # --- pseudo triangles (n_sims, n, n)
    draws = rng.choice(pool, size=(n_sims, n_obs), replace=True)
    r_star = np.zeros((n_sims, n, n))
    r_star[:, obs] = draws
    q_star = np.where(obs, m + r_star * root, 0.0)
    q_star = np.nan_to_num(q_star)
    c_star = np.cumsum(q_star, axis=2)

    # --- refit volume-weighted factors per simulation
    f_star = np.ones((n_sims, n - 1))
    for j in range(n - 1):
        rows = n - 1 - j
        num = c_star[:, :rows, j + 1].sum(axis=1)
        den = c_star[:, :rows, j].sum(axis=1)
        f_star[:, j] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)

    # --- project future incremental means from each pseudo latest diagonal
    reserves = np.zeros((n_sims, n))
    for i in range(1, n):
        a = n - 1 - i
        latest = c_star[:, i, a]
        cum = latest[:, None] * np.cumprod(f_star[:, a:], axis=1)          # ages a+1 .. n-1
        prev = np.concatenate([latest[:, None], cum[:, :-1]], axis=1)
        mu = cum - prev
        if process:
            mu = _odp_process(rng, mu, phi)
        reserves[:, i] = mu.sum(axis=1)

    det = chain_ladder(tri, "volume", tail=1.0).reserve
    return BootstrapResult(reserves, phi, r_adj, det)


def _odp_process(rng: np.random.Generator, mu: np.ndarray, phi: float) -> np.ndarray:
    """Gamma draws with mean |mu| and variance phi |mu|, carrying mu's sign."""
    a = np.abs(mu)
    out = np.zeros_like(mu)
    pos = a > 0
    if phi <= 0:
        return mu
    out[pos] = rng.gamma(shape=a[pos] / phi, scale=phi)
    return np.sign(mu) * out
