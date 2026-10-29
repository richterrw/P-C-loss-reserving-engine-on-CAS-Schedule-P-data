"""Classic textbook triangles used to validate the implementations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .triangle import Triangle

BENCH_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"


def load_benchmark(name: str) -> Triangle:
    """``raa`` (Mack 1993) or ``genins`` (Taylor & Ashe 1983), cumulative."""
    df = pd.read_csv(BENCH_DIR / f"{name}.csv")
    df["lag"] = df["development"] - df["origin"] + 1
    return Triangle.from_long(df, "origin", "lag", "values", name=name)
