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


def hat_diagonal(m: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Leverage of each observed cell in the ODP GLM (log link, origin + age effects).

    H = W^1/2 X (X'WX)^-1 X' W^1/2 with Poisson working weights W = |m|.
    The chain ladder fitted values are the GLM fitted values, so only the
    design matrix is needed.  Corner cells have leverage 1.
    """
    n = m.shape[0]
    cells = np.argwhere(obs)
    X = np.zeros((len(cells), 2 * n - 1))
    X[:, 0] = 1.0
    for r, (i, j) in enumerate(cells):
        if i > 0:
            X[r, i] = 1.0
        if j > 0:
            X[r, n - 1 + j] = 1.0
    w = np.abs(m[obs])
    Xw = X * np.sqrt(w)[:, None]
    h = np.einsum("ij,ij->i", Xw @ np.linalg.pinv(Xw.T @ Xw), Xw)
    out = np.full(m.shape, np.nan)
    out[obs] = np.clip(h, 0.0, 1.0)
    return out


def auto_hetero_groups(n: int, min_cells: int = 10) -> list[list[int]]:
    """Contiguous development-age groups with at least ``min_cells`` residuals each."""
    groups, cur, count = [], [], 0
    for j in range(n):
        cur.append(j)
        count += n - j - (1 if j in (0, n - 1) else 0)  # corner cells carry no residual
        if count >= min_cells:
            groups.append(cur)
            cur, count = [], 0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


def odp_bootstrap(tri: Triangle, n_sims: int = 5000, seed: int | None = 12345, process: bool = True,
                  hat_adjust: bool = False, hetero: bool | list[list[int]] = False) -> BootstrapResult:
    """ODP bootstrap.

    ``hat_adjust`` replaces the degrees-of-freedom factor with the per-cell
    leverage adjustment r / sqrt(1 - h) (Shapland 2016, sec. 3.1).
    ``hetero`` estimates a separate scale parameter for groups of development
    ages, standardises residuals across groups before pooling, and uses each
    group's own scale for process risk (Shapland 2016, sec. 3.4).  Pass a list
    of age-index groups, or True for :func:`auto_hetero_groups`.
    """
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

    # Corner cells are fitted exactly (residual identically zero); keep them out of the pool.
    pool_mask = obs.copy()
    pool_mask[0, n - 1] = pool_mask[n - 1, 0] = False

    if hat_adjust:
        h = hat_diagonal(m, obs)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_adj = np.where(pool_mask & (h < 1 - 1e-9), r / np.sqrt(1 - h), np.nan)
    else:
        r_adj = r * np.sqrt(n_obs / dof)

    # Scale parameter(s): one per development-age group (a single group by default).
    groups = (auto_hetero_groups(n) if hetero is True else hetero) if hetero else [list(range(n))]
    age_group = np.zeros(n, dtype=int)
    for g, ages in enumerate(groups):
        age_group[ages] = g
    cell_group = np.broadcast_to(age_group, (n, n))
    phi_all = float(np.nansum(r**2) / dof)
    phi_g = np.empty(len(groups))
    for g in range(len(groups)):
        sel = pool_mask & (cell_group == g) & np.isfinite(r)
        k = sel.sum()
        phi_g[g] = (np.nansum(r[sel] ** 2) / k) * (n_obs / dof) if k else phi_all
    phi_g = np.where(phi_g > 0, phi_g, phi_all)
    scale = np.sqrt(phi_all / phi_g)          # standardising factor per group
    cell_scale = scale[cell_group]

    pool = (r_adj * cell_scale)[pool_mask & np.isfinite(r_adj)]

    # --- pseudo triangles (n_sims, n, n): residuals de-standardised into each cell's group
    draws = rng.choice(pool, size=(n_sims, n_obs), replace=True)
    r_star = np.zeros((n_sims, n, n))
    r_star[:, obs] = draws
    r_star /= cell_scale
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
            mu = _odp_process(rng, mu, phi_g[age_group[a + 1:]])
        reserves[:, i] = mu.sum(axis=1)

    det = chain_ladder(tri, "volume", tail=1.0).reserve
    return BootstrapResult(reserves, phi_all, r_adj, det)


def _odp_process(rng: np.random.Generator, mu: np.ndarray, phi) -> np.ndarray:
    """Gamma draws with mean |mu| and variance phi |mu|, carrying mu's sign.

    ``phi`` is a scalar or one value per column of ``mu``.
    """
    phi = np.broadcast_to(np.asarray(phi, dtype=float), mu.shape[-1:])
    phi = np.broadcast_to(phi, mu.shape)
    a = np.abs(mu)
    out = np.zeros_like(mu)
    pos = (a > 0) & (phi > 0)
    out[pos] = rng.gamma(shape=a[pos] / phi[pos], scale=phi[pos])
    return np.where(phi > 0, np.sign(mu) * out, mu)
