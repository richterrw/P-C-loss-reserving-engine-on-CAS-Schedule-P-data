"""Expected-loss-ratio methods: Bornhuetter-Ferguson, Cape Cod, Benktander.

All three are credibility blends of the chain ladder ultimate and an a priori
ultimate ``U0 = ELR * premium``.  With ``p = 1 / CDF`` (percent developed):

    chain ladder : U = latest / p                     (credibility Z = 1)
    BF           : U = latest + (1 - p) * U0          (Z = p)
    Benktander   : U = latest + (1 - p) * U_BF        (Z = 1 - (1 - p)^2)

which gives the identity used to explain CL vs BF disagreement:

    U_CL - U_BF = (1 - p) * (U_CL - U0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chainladder import ChainLadderResult


@dataclass
class ELRResult:
    method: str
    latest: np.ndarray
    premium: np.ndarray
    elr: np.ndarray
    cdf: np.ndarray
    ultimate: np.ndarray

    @property
    def prior_ultimate(self) -> np.ndarray:
        return self.elr * self.premium

    @property
    def reserve(self) -> np.ndarray:
        return self.ultimate - self.latest

    @property
    def credibility(self) -> np.ndarray:
        """Weight Z on the chain ladder ultimate implied by this method."""
        cl = self.latest * self.cdf
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (self.ultimate - self.prior_ultimate) / (cl - self.prior_ultimate)
        return z


def cape_cod_elr(latest: np.ndarray, premium: np.ndarray, cdf: np.ndarray) -> float:
    """Stanard-Buhlmann ELR: reported losses over 'used-up' premium."""
    return float(latest.sum() / (premium / cdf).sum())


def bornhuetter_ferguson(cl: ChainLadderResult, premium: np.ndarray, elr, method: str = "BF") -> ELRResult:
    premium = np.asarray(premium, dtype=float)
    elr = np.broadcast_to(np.asarray(elr, dtype=float), premium.shape).copy()
    unreported = 1.0 - 1.0 / cl.cdf
    ult = cl.latest + unreported * elr * premium
    return ELRResult(method, cl.latest, premium, elr, cl.cdf, ult)


def cape_cod(cl: ChainLadderResult, premium: np.ndarray) -> ELRResult:
    elr = cape_cod_elr(cl.latest, np.asarray(premium, float), cl.cdf)
    return bornhuetter_ferguson(cl, premium, elr, method="Cape Cod")


def benktander(cl: ChainLadderResult, premium: np.ndarray, elr, method: str = "Benktander") -> ELRResult:
    bf = bornhuetter_ferguson(cl, premium, elr)
    ult = cl.latest + (1.0 - 1.0 / cl.cdf) * bf.ultimate
    return ELRResult(method, cl.latest, bf.premium, bf.elr, cl.cdf, ult)
