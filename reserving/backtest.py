"""Retrospective test of every method on every clean company in the database."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import LINES, clean_companies
from .engine import BASES, analyze


def run_company(lob: str, code: int, n_sims: int, seed: int) -> dict:
    a = analyze(lob, code, n_sims=n_sims, seed=seed)
    d = a.data
    paid = d.paid.latest.sum()
    actual = a.actual.sum()
    row = {"lob": lob, "grcode": code, "company": d.name, "premium": d.premium.sum(),
           "paid_to_date": paid, "actual_ult": actual, "actual_unpaid": actual - paid}
    for k, v in a.estimates.items():
        row[f"ult::{k}"] = v.sum()

    # Calibration of the variability models, each on its own basis (no tail).
    for b in BASES:
        tri = d.triangle(b)
        latest = tri.latest.sum()
        target = d.actual_ultimate(b).sum()
        m = a.mack[b]
        mu, sig = m.lognormal_params_ultimate(latest)
        row[f"mack_pct::{b}"] = stats.lognorm.cdf(target, s=sig, scale=np.exp(mu))
        row[f"mack_cv::{b}"] = m.total_se / (latest + m.reserve.sum())
        row[f"boot_pct::{b}"] = a.boot[b].rank_of(target - latest)
        row[f"boot_cv::{b}"] = a.boot[b].total_se / (latest + a.boot[b].total.mean())
        row[f"cl_ult::{b}"] = latest + m.reserve.sum()
        row[f"target::{b}"] = target
    return row


def run(lines=None, n_sims: int = 2000, seed: int = 7) -> pd.DataFrame:
    rows = []
    for lob in lines or LINES:
        for code in clean_companies(lob):
            rows.append(run_company(lob, code, n_sims, seed))
    return pd.DataFrame(rows)


def method_columns(df: pd.DataFrame) -> list[str]:
    return [c.split("::", 1)[1] for c in df.columns if c.startswith("ult::")]


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per line and method: dollar-weighted bias and median absolute error.

    Errors are expressed as a share of the actual unpaid amount, i.e. how wrong
    the 1997 reserve would have been.  ``best`` counts how often the method was
    the closest of all point estimates for a company.
    """
    methods = method_columns(df)
    err = pd.DataFrame({m: df[f"ult::{m}"] - df["actual_ult"] for m in methods})
    rel = err.div(df["actual_unpaid"], axis=0)
    closest = err.abs().idxmin(axis=1)
    out = []
    for lob, idx in df.groupby("lob").groups.items():
        for m in methods:
            out.append({
                "lob": lob, "method": m, "n": len(idx),
                "bias_pct": err.loc[idx, m].sum() / df.loc[idx, "actual_unpaid"].sum(),
                "median_abs_pct": rel.loc[idx, m].abs().median(),
                "best_share": (closest.loc[idx] == m).mean(),
            })
    return pd.DataFrame(out)


def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """How often the realised outcome falls inside each model's intervals.

    A well calibrated model puts ~10% of outcomes below its 10th percentile,
    ~80% inside the 10-90 band, and passes a Kolmogorov-Smirnov test that
    predictive percentiles are uniform (the approach of Meyers 2015).
    ``median_cv`` (s.e. / mean ultimate) measures sharpness: a model can
    look calibrated simply by being uninformatively wide.
    """
    out = []
    for model in ("mack", "boot"):
        for b in BASES:
            p = df[f"{model}_pct::{b}"].dropna().to_numpy()
            out.append({
                "model": {"mack": "Mack (lognormal)", "boot": "ODP bootstrap"}[model], "basis": b, "n": len(p),
                "below_p10": (p < 0.10).mean(), "inside_p10_p90": ((p >= 0.10) & (p <= 0.90)).mean(),
                "above_p90": (p > 0.90).mean(), "median_pct": np.median(p),
                "ks_pvalue": stats.kstest(p, "uniform").pvalue,
                "median_cv": df[f"{model}_cv::{b}"].abs().median(),
            })
    return pd.DataFrame(out)
