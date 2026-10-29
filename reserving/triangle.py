"""Loss development triangle container.

A triangle is stored as a square ``n x n`` array of *cumulative* values with
origin (accident year) on the rows and development age on the columns.  Cells
that are not yet observed at the valuation date (below the latest diagonal)
are ``NaN``.  Origin ``i`` (0-based) has ``n - i`` observed development ages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Triangle:
    values: np.ndarray
    origins: tuple[int, ...]
    name: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        if v.ndim != 2 or v.shape[0] != v.shape[1]:
            raise ValueError("triangle must be a square 2-D array")
        if len(self.origins) != v.shape[0]:
            raise ValueError("one origin label per row is required")
        object.__setattr__(self, "values", v)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_square(cls, square: np.ndarray, origins, name: str = "") -> "Triangle":
        """Mask a fully developed square down to the upper-left triangle."""
        sq = np.asarray(square, dtype=float).copy()
        sq[~upper_mask(sq.shape[0])] = np.nan
        return cls(sq, tuple(origins), name)

    @classmethod
    def from_incremental(cls, incremental: np.ndarray, origins, name: str = "") -> "Triangle":
        inc = np.asarray(incremental, dtype=float)
        cum = np.cumsum(np.where(np.isnan(inc), 0.0, inc), axis=1)
        return cls.from_square(cum, origins, name)

    @classmethod
    def from_long(cls, df: pd.DataFrame, origin: str, dev: str, value: str, name: str = "") -> "Triangle":
        """Build from long format where ``dev`` is a 1-based development lag."""
        wide = df.pivot_table(index=origin, columns=dev, values=value, aggfunc="sum")
        wide = wide.sort_index().sort_index(axis=1)
        return cls.from_square(wide.to_numpy(), wide.index.tolist(), name)

    # ------------------------------------------------------------- properties
    @property
    def n(self) -> int:
        return self.values.shape[0]

    @property
    def mask(self) -> np.ndarray:
        return upper_mask(self.n)

    @property
    def latest(self) -> np.ndarray:
        """Latest diagonal (most recent cumulative value per origin)."""
        return np.array([self.values[i, self.n - 1 - i] for i in range(self.n)])

    @property
    def latest_age_index(self) -> np.ndarray:
        return np.arange(self.n)[::-1]

    @property
    def incremental(self) -> np.ndarray:
        inc = np.diff(self.values, axis=1, prepend=0.0)
        inc[~self.mask] = np.nan
        return inc

    def link_ratios(self) -> np.ndarray:
        """Age-to-age factors C[i, j+1] / C[i, j]; shape ``(n, n-1)``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.values[:, 1:] / self.values[:, :-1]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=list(self.origins), columns=range(1, self.n + 1))


def upper_mask(n: int) -> np.ndarray:
    """Boolean mask of observed cells: ``i + j <= n - 1``."""
    i, j = np.indices((n, n))
    return i + j <= n - 1
