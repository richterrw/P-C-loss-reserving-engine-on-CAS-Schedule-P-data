"""Retrospective test of every method on every clean company in the database."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import LINES, clean_companies, company
from .engine import BASES, analyze
from .chainladder import development_factors
from .tail import CURVES, curve_extrapolation_test, tail_factor
from .triangle import Triangle


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
        sh = a.boot_shapland[b]
        row[f"shap_pct::{b}"] = sh.rank_of(target - latest)
        row[f"shap_cv::{b}"] = sh.total_se / (latest + sh.total.mean())
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
    names = {"mack": "Mack (lognormal)", "boot": "ODP bootstrap", "shap": "ODP bootstrap (Shapland)"}
    for model in names:
        for b in BASES:
            p = df[f"{model}_pct::{b}"].dropna().to_numpy()
            out.append({
                "model": names[model], "basis": b, "n": len(p),
                "below_p10": (p < 0.10).mean(), "inside_p10_p90": ((p >= 0.10) & (p <= 0.90)).mean(),
                "above_p90": (p > 0.90).mean(), "median_pct": np.median(p),
                "ks_pvalue": stats.kstest(p, "uniform").pvalue,
                "median_cv": df[f"{model}_cv::{b}"].abs().median(),
            })
    return pd.DataFrame(out)


def tail_study(lines=None, k: int = 5) -> pd.DataFrame:
    """Curve extrapolation test on every clean company's paid triangle.

    Paid only: incurred development past age 6 hovers around 1.0 (often
    below), so a relative error on (f - 1) is meaningless there.

    Error is measured on the development portion: (predicted - actual) / (actual - 1)
    for the cumulative factor from age ``k + 1`` to 10.
    """
    rows = []
    for lob in lines or LINES:
        for code in clean_companies(lob):
            cd = company(lob, code)
            t = curve_extrapolation_test(cd.paid, k=k)
            row = {"lob": lob, "grcode": code, "actual": t["actual"]}
            for c in CURVES:
                row[c] = t[c]
                row[f"err::{c}"] = (t[c] - t["actual"]) / (t["actual"] - 1) if t["actual"] > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def industry_tails(lines=None, curve: str = "exponential", start: int = 2) -> pd.DataFrame:
    """How much development each line still has beyond age 10, on industry aggregates.

    ``paid_tail_curve`` extrapolates the aggregate paid factors with ``curve``;
    ``incurred_over_paid_10`` is the observed gap between case incurred and
    paid at age 10 for AY 1988 (outstanding case reserves).  ``bulk_ibnr_10``
    is the IBNR companies still carried at age 10, as a share of case incurred.
    """
    rows = []
    for lob in lines or LINES:
        P = I = B = None
        for code in clean_companies(lob):
            cd = company(lob, code)
            p, i, b = (np.nan_to_num(x) for x in (cd.paid_square, cd.incurred_square, cd.booked_square))
            P, I, B = (p, i, b) if P is None else (P + p, I + i, B + b)
        paid_tri = Triangle.from_square(P, cd.origins)
        f = development_factors(paid_tri)
        rows.append({
            "lob": lob,
            "paid_tail_curve": tail_factor(f, curve, start),
            "incurred_over_paid_10": I[0, -1] / P[0, -1],
            "bulk_ibnr_10": B[0, -1] / I[0, -1] - 1,
        })
    return pd.DataFrame(rows).set_index("lob")
