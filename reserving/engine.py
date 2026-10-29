"""Run every method on one company and line, on a common footing.

Point estimates all target the same quantity: ultimate loss for accident
years 1988-1997, measured against *case incurred at lag 10* (the realised
outcome).  Paid methods therefore need a paid-to-incurred tail; we use the
incurred/paid ratio of the oldest accident year at lag 10, which is on the
1997 diagonal and so known at the valuation date.

Variability models (Mack, ODP bootstrap) are judged against the quantity they
model: the lag-10 value on their own basis, with no tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from .bf import benktander, bornhuetter_ferguson, cape_cod
from .bootstrap import BootstrapResult, odp_bootstrap
from .chainladder import ChainLadderResult, chain_ladder
from .data import CompanyData, clean_companies, company
from .mack import MackResult, mack
from .munich import MunichResult, munich_chain_ladder
from .triangle import Triangle

BASES = ("paid", "incurred")


@lru_cache(maxsize=None)
def industry_elr(lob: str) -> np.ndarray:
    """A priori ELR by accident year from the line's industry aggregate.

    Sums the incurred triangles and premiums of every clean company in the
    line and runs a chain ladder on the aggregate.  Uses only data known at
    the 1997 valuation and captures the underwriting cycle by accident year.
    """
    codes = clean_companies(lob)
    tri = prem = None
    for c in codes:
        cd = company(lob, c)
        t = np.nan_to_num(cd.incurred.values)
        tri = t if tri is None else tri + t
        prem = cd.premium if prem is None else prem + cd.premium
    agg = Triangle.from_square(tri, cd.origins, f"{lob} industry")
    return chain_ladder(agg).ultimate / prem


def paid_tail(cd: CompanyData) -> float:
    """Incurred-to-paid ratio of the oldest accident year at its latest age."""
    return max(1.0, cd.incurred.latest[0] / cd.paid.latest[0])


@dataclass
class CompanyAnalysis:
    data: CompanyData
    cl: dict[str, ChainLadderResult]
    estimates: dict[str, np.ndarray]           # method label -> ultimate by AY
    prior_elr: np.ndarray
    munich: MunichResult | None = None
    mack: dict[str, MackResult] = field(default_factory=dict)
    boot: dict[str, BootstrapResult] = field(default_factory=dict)
    boot_shapland: dict[str, BootstrapResult] = field(default_factory=dict)

    @property
    def actual(self) -> np.ndarray:
        return self.data.actual_ultimate("incurred")

    def by_origin(self) -> pd.DataFrame:
        d = self.data
        df = pd.DataFrame({
            "premium": d.premium,
            "paid": d.paid.latest,
            "case_incurred": d.incurred.latest,
            "pct_dev_paid": 1 / self.cl["paid"].cdf,
            "pct_dev_incurred": 1 / self.cl["incurred"].cdf,
            "prior_elr": self.prior_elr,
        }, index=pd.Index(d.origins, name="AY"))
        for k, v in self.estimates.items():
            df[k] = v
        df["Actual (lag 10)"] = self.actual
        return df

    def summary(self) -> pd.DataFrame:
        paid = self.data.paid.latest.sum()
        actual = self.actual.sum()
        rows = []
        for k, v in self.estimates.items():
            ult = v.sum()
            rows.append({
                "method": k, "ultimate": ult, "unpaid": ult - paid,
                "error": ult - actual, "error_pct_of_unpaid": (ult - actual) / (actual - paid),
            })
        rows.append({"method": "Actual (lag 10)", "ultimate": actual, "unpaid": actual - paid,
                     "error": 0.0, "error_pct_of_unpaid": 0.0})
        return pd.DataFrame(rows).set_index("method")


def point_estimates(cd: CompanyData, prior_elr: np.ndarray) -> tuple[dict, dict, MunichResult]:
    tail = {"paid": paid_tail(cd), "incurred": 1.0}
    cls = {b: chain_ladder(cd.triangle(b), tail=tail[b]) for b in BASES}
    est = {}
    for b in BASES:
        cl, tag = cls[b], b.capitalize()
        est[f"CL {tag}"] = cl.ultimate
        est[f"BF {tag}"] = bornhuetter_ferguson(cl, cd.premium, prior_elr).ultimate
        est[f"Cape Cod {tag}"] = cape_cod(cl, cd.premium).ultimate
        est[f"Benktander {tag}"] = benktander(cl, cd.premium, prior_elr).ultimate
    # Paid/incurred reconciling methods (section 4 of the README).
    mcl = munich_chain_ladder(cd.paid, cd.incurred)
    est["MCL Paid"] = mcl.paid_ultimate * tail["paid"]
    est["MCL Incurred"] = mcl.incurred_ultimate
    est["Company booked"] = cd.booked_ultimate
    return cls, est, mcl


def analyze(lob: str, grcode: int, n_sims: int = 5000, seed: int = 2024, variability: bool = True) -> CompanyAnalysis:
    cd = company(lob, grcode)
    prior = industry_elr(lob)
    cls, est, mcl = point_estimates(cd, prior)
    out = CompanyAnalysis(cd, cls, est, prior, mcl)
    if variability:
        for b in BASES:
            out.mack[b] = mack(cd.triangle(b))
            out.boot[b] = odp_bootstrap(cd.triangle(b), n_sims=n_sims, seed=seed)
            out.boot_shapland[b] = odp_bootstrap(cd.triangle(b), n_sims=n_sims, seed=seed,
                                                 hat_adjust=True, hetero=True)
    return out
