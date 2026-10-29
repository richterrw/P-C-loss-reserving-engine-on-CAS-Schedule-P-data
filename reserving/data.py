"""Loader for the CAS Loss Reserving Database (NAIC Schedule P, Meyers & Shi).

Each line-of-business file holds a full 10 x 10 square per company for
accident years 1988-1997 and development lags 1-10.  At the 1997 valuation
only the upper triangle was known; the lower triangle (calendar years
1998-2006) is the realised outcome we backtest against.

Field conventions follow Meyers (2015, CAS Monograph 1):
  * paid      = CumPaidLoss                     (Schedule P Part 3)
  * incurred  = IncurLoss - BulkLoss            (case incurred: paid + case reserves)
  * booked    = IncurLoss                       (company's own ultimate incl. IBNR)
  * premium   = EarnedPremNet
All amounts are net of reinsurance, in $000s, and include DCC (defence) expense.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .triangle import Triangle

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LINES = {
    "wkcomp": "Workers' Compensation",
    "ppauto": "Private Passenger Auto Liability",
    "comauto": "Commercial Auto Liability",
    "medmal": "Medical Malpractice",
    "prodliab": "Products Liability",
    "othliab": "Other Liability",
}

_FIELDS = ["IncurLoss", "CumPaidLoss", "BulkLoss", "EarnedPremDIR", "EarnedPremCeded", "EarnedPremNet", "PostedReserve97"]


@lru_cache(maxsize=None)
def load_line(lob: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{lob}_pos.csv")
    rename = {c: f for c in df.columns for f in _FIELDS if c.startswith(f + "_")}
    df = df.rename(columns=rename)
    df["CaseIncurred"] = df["IncurLoss"] - df["BulkLoss"]
    df["LOB"] = lob
    return df.sort_values(["GRCODE", "AccidentYear", "DevelopmentLag"]).reset_index(drop=True)


def _square(g: pd.DataFrame, field: str) -> np.ndarray:
    return g.pivot(index="AccidentYear", columns="DevelopmentLag", values=field).to_numpy(float)


@dataclass
class CompanyData:
    lob: str
    grcode: int
    name: str
    origins: tuple[int, ...]
    premium: np.ndarray
    paid_square: np.ndarray
    incurred_square: np.ndarray
    booked_square: np.ndarray

    @property
    def paid(self) -> Triangle:
        return Triangle.from_square(self.paid_square, self.origins, "paid")

    @property
    def incurred(self) -> Triangle:
        return Triangle.from_square(self.incurred_square, self.origins, "incurred")

    def triangle(self, basis: str) -> Triangle:
        return self.paid if basis == "paid" else self.incurred

    def square(self, basis: str) -> np.ndarray:
        return self.paid_square if basis == "paid" else self.incurred_square

    @property
    def paid_to_date(self) -> np.ndarray:
        return self.paid.latest

    @property
    def booked_ultimate(self) -> np.ndarray:
        """Company's own 1997 booked ultimate (case incurred + bulk/IBNR)."""
        return Triangle.from_square(self.booked_square, self.origins).latest

    def actual_ultimate(self, basis: str = "incurred") -> np.ndarray:
        """Value at lag 10 -- the realised outcome as of calendar year 2006."""
        return self.square(basis)[:, -1]


def company(lob: str, grcode: int) -> CompanyData:
    df = load_line(lob)
    g = df[df.GRCODE == grcode]
    if len(g) != 100:
        raise KeyError(f"GRCODE {grcode} not found (or incomplete) in {lob}")
    prem = g[g.DevelopmentLag == 1].sort_values("AccidentYear")["EarnedPremNet"].to_numpy(float)
    return CompanyData(
        lob, int(grcode), str(g.GRNAME.iloc[0]).strip(), tuple(sorted(g.AccidentYear.unique())),
        prem, _square(g, "CumPaidLoss"), _square(g, "CaseIncurred"), _square(g, "IncurLoss"),
    )


def clean_companies(lob: str, min_premium: float = 1_000.0) -> list[int]:
    """Companies whose triangles support a sensible test.

    Requires, for every accident year: positive net earned premium, positive
    cumulative paid and case incurred in every *observed* cell, a positive
    realised lag-10 value on both bases, total net premium of at least
    ``min_premium`` ($000s) across the ten years, and some post-1997 movement
    (actual unpaid >= 1% of actual ultimate).  This screens out shells and
    data errors such as negative cumulative paid or frozen later diagonals.
    """
    df = load_line(lob)
    keep = []
    for code, g in df.groupby("GRCODE"):
        if len(g) != 100:
            continue
        up = g[g.AccidentYear + g.DevelopmentLag - 1 <= 1997]
        prem = g[g.DevelopmentLag == 1].EarnedPremNet
        if (prem <= 0).any() or prem.sum() < min_premium:
            continue
        if (up.CumPaidLoss <= 0).any() or (up.CaseIncurred <= 0).any():
            continue
        last = g[g.DevelopmentLag == 10]
        if (last.CumPaidLoss <= 0).any() or (last.CaseIncurred <= 0).any():
            continue
        # Frozen post-1997 data (no payments or case movement after the
        # valuation date) is a reporting artefact, not a real outcome.
        diag = up[up.AccidentYear + up.DevelopmentLag - 1 == 1997]
        if last.CaseIncurred.sum() - diag.CumPaidLoss.sum() < 0.01 * last.CaseIncurred.sum():
            continue
        keep.append(int(code))
    return keep
