"""Build reports/REPORT.md, figures and backtest CSVs from the raw data.

    python scripts/build_report.py            # full run (~1 minute)
    python scripts/build_report.py --sims 500 # quicker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reserving import backtest as bt  # noqa: E402
from reserving import diagnostics as dg  # noqa: E402
from reserving.benchmarks import BENCH_DIR, load_benchmark  # noqa: E402
from reserving.bootstrap import fitted_incrementals, hat_diagonal, odp_bootstrap  # noqa: E402
from reserving.chainladder import chain_ladder  # noqa: E402
from reserving.data import LINES  # noqa: E402
from reserving.engine import analyze  # noqa: E402
from reserving.mack import mack  # noqa: E402
from reserving.munich import munich_chain_ladder  # noqa: E402
from reserving.triangle import Triangle, upper_mask  # noqa: E402

OUT = ROOT / "reports"
FIG = OUT / "figures"

SHOWCASES = [
    ("othliab", 1767, "Methods disagree: long tail, a shifting payment pattern"),
    ("medmal", 669, "Methods disagree, and the company was further off than any of them"),
    ("ppauto", 1767, "Methods agree on the point estimate; the variability models are still too confident"),
]

# Reference categorical palette (light mode), fixed order; ink for the actual outcome.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 130,
})


def money(x: float) -> str:
    return f"{x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:+.{d}f}%"


def md_table(df: pd.DataFrame, fmt=None, index=True) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        f = fmt.get(c)
        if f:
            d[c] = d[c].map(f)
    return d.to_markdown(index=index)


# ---------------------------------------------------------------- benchmarks
def benchmark_section() -> str:
    raa, gen = load_benchmark("raa"), load_benchmark("genins")
    rm, gm = mack(raa), mack(gen)
    gb = odp_bootstrap(gen, n_sims=10_000, seed=1)
    d = pd.read_csv(BENCH_DIR / "mcl.csv")
    d["lag"] = d.development - d.origin + 1
    mcl = munich_chain_ladder(Triangle.from_long(d, "origin", "lag", "paid"), Triangle.from_long(d, "origin", "lag", "incurred"))
    rows = [
        ("RAA", "Chain ladder reserve", 52_135, chain_ladder(raa).reserve.sum(), "Mack (1993)"),
        ("RAA", "Mack total s.e.", 26_909, rm.total_se, "Mack (1993); R ChainLadder"),
        ("RAA", "Mack s.e., AY 1990", 24_566, rm.se[-1], "R ChainLadder"),
        ("Taylor-Ashe", "Chain ladder reserve", 18_680_856, chain_ladder(gen).reserve.sum(), "Mack (1993)"),
        ("Taylor-Ashe", "Mack total s.e.", 2_447_095, gm.total_se, "Mack (1993); R ChainLadder"),
        ("Taylor-Ashe", "ODP scale parameter phi", 52_601, gb.phi, "England & Verrall (2002)"),
        ("Taylor-Ashe", "Bootstrap prediction error / reserve", "~16%", f"{gb.total_se / gb.total.mean():.1%}", "England & Verrall (2002)"),
        ("Taylor-Ashe", "ODP GLM hat-matrix trace (= parameters)", 19, np.nansum(hat_diagonal(fitted_incrementals(gen), upper_mask(gen.n))), "Shapland (2016)"),
        ("Quarg-Mack", "MCL lambda paid / incurred", "0.6360 / 0.4362", f"{mcl.lambda_paid:.4f} / {mcl.lambda_incurred:.4f}", "chainladder-python MunichAdjustment"),
        ("Quarg-Mack", "MCL ultimate paid", 32_444, mcl.paid_ultimate.sum(), "chainladder-python MunichAdjustment"),
        ("Quarg-Mack", "MCL ultimate incurred", 32_708, mcl.incurred_ultimate.sum(), "chainladder-python MunichAdjustment"),
    ]
    df = pd.DataFrame(rows, columns=["triangle", "quantity", "published", "this engine", "source"])
    for c in ("published", "this engine"):
        df[c] = df[c].map(lambda v: money(v) if isinstance(v, (int, float, np.floating)) else v)
    cy, co = dg.calendar_year_test(raa), dg.correlation_test(raa)
    return (
        "## 1. Validation against published results\n\n"
        "Before touching Schedule P, every method is checked against textbook triangles "
        "with known answers (also enforced in `tests/`).\n\n"
        + df.to_markdown(index=False)
        + f"\n\nMack's (1994) assumption tests on RAA: calendar-year statistic Z = {cy['Z']:.0f} "
        f"(E[Z] = {cy['E[Z]']:.3f}); adjacent-factor correlation T = {co['T']:.3f} "
        f"(50% band ±{co['band50'][1]:.3f}), matching the paper.\n"
    )


# ----------------------------------------------------------------- showcases
POINT_METHODS = ["CL Paid", "CL Incurred", "BF Paid", "BF Incurred", "Cape Cod Incurred", "MCL Paid"]


def fig_ultimates(a, path: Path) -> None:
    d = a.data
    yrs = np.array(d.origins)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for k, col in zip(POINT_METHODS, C):
        ax.plot(yrs, a.estimates[k] / d.premium, color=col, label=k, marker="o", ms=3.5)
    ax.plot(yrs, a.actual / d.premium, color=INK, lw=2.6, label="Actual (lag 10)", marker="s", ms=4)
    ax.plot(yrs, a.prior_elr, color=MUTED, lw=1.2, ls=(0, (1, 2)), label="Industry prior ELR")
    ax.set_ylabel("Ultimate loss ratio")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"{d.name} - {LINES[d.lob]}: ultimate loss ratio by accident year")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.set_xticks(yrs)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_gap(gap: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    yrs = gap.index.to_numpy()
    ax.bar(yrs, gap["cl_minus_bf"], color=C[0], width=0.7)
    for x, y, u in zip(yrs, gap["cl_minus_bf"], gap["pct_unreported"]):
        ax.annotate(f"{u:.0%}", (x, y), textcoords="offset points", xytext=(0, 3 if y >= 0 else -11),
                    ha="center", fontsize=7.5, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(yrs)
    ax.set_ylabel("CL - BF ultimate ($000)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(title)
    ax.text(0.0, -0.22, "Labels: share of ultimate still unreported (1 - 1/CDF), the leverage on each year's surprise.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_paid_incurred(ratios: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for age, col in zip((1, 2, 3, 4), C):
        s = ratios[age].dropna()
        ax.plot(s.index, s.values, color=col, marker="o", ms=3.5, label=f"age {age}")
    ax.set_ylabel("Paid / case incurred")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks(ratios.index)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_distribution(a, basis: str, path: Path) -> None:
    d = a.data
    b, m = a.boot[basis], a.mack[basis]
    latest = d.triangle(basis).latest.sum()
    actual = d.actual_ultimate(basis).sum() - latest
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.hist(b.total, bins=70, color=C[0], alpha=0.55, density=True, label="ODP bootstrap", edgecolor=SURFACE, linewidth=0.3)
    mu, sig = m.lognormal_params_ultimate(latest)
    lo, hi = np.percentile(b.total, [0.1, 99.9])
    x = np.linspace(lo, hi, 400)
    ax.plot(x, stats.lognorm.pdf(x + latest, s=sig, scale=np.exp(mu)), color=C[1], label="Mack (lognormal)")
    pad = 0.06 * (hi - lo)
    ax.set_xlim(min(lo, actual) - pad, max(hi, actual) + pad)
    ax.axvline(actual, color=INK, lw=2, label=f"Actual: {money(actual)}")
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Total {basis} development to lag 10 ($000)")
    ax.set_title(f"{d.name} - {LINES[d.lob]}: predictive distribution, {basis} basis")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def showcase_section(i: int, lob: str, code: int, headline: str, sims: int) -> str:
    a = analyze(lob, code, n_sims=sims, seed=2024)
    d = a.data
    slug = f"{lob}_{code}"
    fig_ultimates(a, FIG / f"{slug}_ultimates.png")

    paid_cl = a.cl["paid"]
    gap = dg.cl_bf_gap(paid_cl.ultimate, a.prior_elr * d.premium, paid_cl.cdf, d.origins)
    inc_cl = a.cl["incurred"]
    gap_i = dg.cl_bf_gap(inc_cl.ultimate, a.prior_elr * d.premium, inc_cl.cdf, d.origins)
    fig_gap(gap, f"{d.name} - {LINES[d.lob]}: why paid CL and paid BF differ", FIG / f"{slug}_gap.png")
    ratios = dg.paid_to_incurred(d.paid, d.incurred)
    fig_paid_incurred(ratios, f"{d.name} - {LINES[d.lob]}: paid-to-incurred ratio at fixed ages", FIG / f"{slug}_paid_incurred.png")
    fig_distribution(a, "paid", FIG / f"{slug}_dist_paid.png")

    s = a.summary()
    s["share of actual unpaid"] = s["unpaid"] / s.loc["Actual (lag 10)", "unpaid"]
    tbl = md_table(s[["ultimate", "unpaid", "error", "error_pct_of_unpaid"]].rename(columns={"error_pct_of_unpaid": "error / actual unpaid"}),
                   {"ultimate": money, "unpaid": money, "error": money, "error / actual unpaid": pct})

    gap_recent = gap["cl_minus_bf"].iloc[-3:].sum() / gap["cl_minus_bf"].sum() if gap["cl_minus_bf"].sum() else np.nan
    trend = dg.ratio_trend(ratios)
    cy = {b: dg.calendar_year_test(d.triangle(b)) for b in ("paid", "incurred")}
    co = {b: dg.correlation_test(d.triangle(b)) for b in ("paid", "incurred")}
    ave = {b: dg.actual_vs_expected_diagonal(d.triangle(b)) for b in ("paid", "incurred")}
    ptrend = d.premium[-1] / d.premium[0] - 1

    var_rows = []
    for b in ("paid", "incurred"):
        m, bs = a.mack[b], a.boot[b]
        latest = d.triangle(b).latest.sum()
        act = d.actual_ultimate(b).sum() - latest
        mu, sig = m.lognormal_params_ultimate(latest)
        var_rows.append({
            "basis": b, "CL development": m.reserve.sum(), "Mack s.e.": m.total_se,
            "bootstrap mean": bs.total.mean(), "bootstrap s.e.": bs.total_se,
            "boot 5th-95th": f"{money(bs.percentile(5))} - {money(bs.percentile(95))}",
            "actual": act, "actual pct (Mack)": stats.lognorm.cdf(act + latest, s=sig, scale=np.exp(mu)),
            "actual pct (boot)": bs.rank_of(act),
        })
    vt = md_table(pd.DataFrame(var_rows).set_index("basis"), {
        "CL development": money, "Mack s.e.": money, "bootstrap mean": money, "bootstrap s.e.": money,
        "actual": money, "actual pct (Mack)": lambda v: f"{v:.0%}", "actual pct (boot)": lambda v: f"{v:.0%}"})

    gap_tbl = md_table(pd.DataFrame({
        "% unreported (paid)": gap["pct_unreported"], "paid CL - BF": gap["cl_minus_bf"],
        "% unreported (inc)": gap_i["pct_unreported"], "incurred CL - BF": gap_i["cl_minus_bf"],
        "actual ult": a.actual,
    }), {"% unreported (paid)": lambda v: f"{v:.0%}", "% unreported (inc)": lambda v: f"{v:.0%}",
         "paid CL - BF": money, "incurred CL - BF": money, "actual ult": money})

    def ave_line(b):
        e = ave[b]
        return (f"{b}: latest diagonal came in at {e['actual'].sum() / e['expected'].sum():.3f}x what the 1996 factors "
                f"predicted (by AY: {', '.join(f'{v:.2f}' for v in e['a_over_e'])})")

    tr_lines = "; ".join(f"age {int(k)}: {r.first:.0%} -> {r.last:.0%} (p = {r.p_value:.2f})" for k, r in trend.iterrows())
    facts = [
        f"Premium changed {pct(ptrend, 0)} from AY {d.origins[0]} to AY {d.origins[-1]}.",
        f"{gap_recent:.0%} of the paid CL - BF gap sits in the three most recent accident years.",
        f"Paid-to-incurred ratio at fixed ages, oldest to newest AY: {tr_lines}.",
        "Calendar-year effect test (Mack 1994): "
        + "; ".join(f"{b} Z = {cy[b]['Z']:.0f} vs 95% band [{cy[b]['ci95'][0]:.1f}, {cy[b]['ci95'][1]:.1f}]"
                    f"{' **flagged**' if cy[b]['calendar_effect'] else ''}" for b in cy) + ".",
        "Adjacent-factor correlation test: "
        + "; ".join(f"{b} T = {co[b]['T']:.2f}{' **flagged**' if co[b]['correlated'] else ''}" for b in co) + ".",
        "Actual vs expected on the 1997 diagonal - " + ave_line("paid") + "; " + ave_line("incurred") + ".",
        f"Paid tail used to reach incurred: {a.cl['paid'].tail:.3f} (AY {d.origins[0]} incurred/paid at age 10).",
    ]
    return (
        f"### 2.{i} {d.name} - {LINES[lob]} (NAIC group {code})\n\n*{headline}.*\n\n"
        f"Net earned premium AY 1988-97: {money(d.premium.sum())}k. Paid to date at 12/1997: {money(d.paid.latest.sum())}k.\n\n"
        f"![ultimates](figures/{slug}_ultimates.png)\n\n"
        f"**Point estimates (all measured against case incurred at lag 10, $000)**\n\n{tbl}\n\n"
        f"**Where the disagreement comes from, by accident year**\n\n{gap_tbl}\n\n"
        f"![gap](figures/{slug}_gap.png)\n\n![paid/incurred](figures/{slug}_paid_incurred.png)\n\n"
        "**Diagnostics**\n\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"
        f"**Variability (each model judged on its own basis, lag 10, no tail)**\n\n{vt}\n\n"
        f"![distribution](figures/{slug}_dist_paid.png)\n"
    )


# ------------------------------------------------------------------ backtest
def fig_calibration(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    series = [("mack_pct::paid", "Mack, paid"), ("mack_pct::incurred", "Mack, incurred"),
              ("boot_pct::paid", "Bootstrap, paid"), ("boot_pct::incurred", "Bootstrap, incurred")]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, label="Perfect calibration")
    n = len(df)
    lo, hi = np.linspace(0, 1, 101), None
    band = 1.36 / np.sqrt(n)  # 95% Kolmogorov-Smirnov band
    ax.fill_between(lo, np.clip(lo - band, 0, 1), np.clip(lo + band, 0, 1), color=GRID, alpha=0.8, lw=0, label="95% KS band")
    for (col, lab), c in zip(series, C):
        p = np.sort(df[col].dropna().to_numpy())
        ax.plot(p, np.arange(1, len(p) + 1) / len(p), color=c, label=lab)
    ax.set_xlabel("Predicted percentile of the actual outcome")
    ax.set_ylabel("Share of companies at or below")
    ax.set_title(f"Calibration across {n} companies (P-P plot)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_errors(df: pd.DataFrame, path: Path) -> None:
    methods = ["CL Paid", "BF Paid", "Cape Cod Paid", "CL Incurred", "BF Incurred", "Cape Cod Incurred",
               "Benktander Incurred", "MCL Paid", "MCL Incurred", "Company booked"]
    err = [((df[f"ult::{m}"] - df.actual_ult) / df.actual_unpaid).clip(-1.5, 1.5) for m in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bp = ax.boxplot(err, orientation="horizontal", tick_labels=methods, showfliers=False, widths=0.55, patch_artist=True,
                    medianprops={"color": INK, "lw": 1.8}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch in bp["boxes"]:
        patch.set(facecolor=C[0], alpha=0.35, edgecolor=C[0])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.invert_yaxis()
    ax.set_xlabel("Reserve error as % of actual unpaid (box = middle 50%, whiskers = 1.5 IQR)")
    ax.set_title(f"Reserve error by method, {len(df)} companies, all six lines")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def backtest_section(sims: int) -> str:
    df = bt.run(n_sims=sims)
    df.to_csv(OUT / "backtest_companies.csv", index=False)
    acc = bt.accuracy_table(df)
    cal = bt.calibration_table(df)
    acc.to_csv(OUT / "backtest_accuracy.csv", index=False)
    cal.to_csv(OUT / "backtest_calibration.csv", index=False)
    fig_calibration(df, FIG / "backtest_calibration.png")
    fig_errors(df, FIG / "backtest_errors.png")

    order = ["CL Paid", "BF Paid", "Cape Cod Paid", "Benktander Paid", "CL Incurred", "BF Incurred",
             "Cape Cod Incurred", "Benktander Incurred", "MCL Paid", "MCL Incurred", "Company booked"]
    counts = df.groupby("lob").size()
    cols = [f"{l} (n={counts[l]})" for l in LINES]

    def pivot(val):
        p = acc.pivot(index="method", columns="lob", values=val).loc[order, list(LINES)]
        p.columns = cols
        return p

    methods = bt.method_columns(df)
    rel = pd.DataFrame({m: (df[f"ult::{m}"] - df.actual_ult) / df.actual_unpaid for m in methods})
    overall = pd.DataFrame({
        "median error": rel.median(), "median |error|": rel.abs().median(),
        "within ±20%": (rel.abs() <= 0.2).mean(),
        "closest estimate": rel.abs().idxmin(axis=1).value_counts(normalize=True).reindex(methods).fillna(0),
    }).loc[order]
    f = lambda v: f"{v:.0%}"  # noqa: E731
    fs = lambda v: f"{v:+.0%}"  # noqa: E731
    cal_fmt = cal.set_index(["model", "basis"])
    return (
        "## 3. Backtest: every clean company in the database\n\n"
        f"{len(df)} company/line combinations pass the data screen (`reserving.data.clean_companies`). "
        "For each one the 1997 upper triangles are run through every method and the estimate is compared "
        "with the realised lag-10 outcome. Errors are quoted as a share of the actual unpaid amount, i.e. "
        "*how wrong the reserve would have been*.\n\n"
        "**All lines pooled (equal weight per company)**\n\n"
        + md_table(overall, {"median error": fs, "median |error|": f, "within ±20%": f, "closest estimate": f})
        + "\n\n![errors](figures/backtest_errors.png)\n\n"
        "**Dollar-weighted bias by line** (sum of errors / sum of actual unpaid; positive = over-reserved)\n\n"
        + md_table(pivot("bias_pct"), {c: fs for c in cols})
        + "\n\n**Median absolute error by line**\n\n"
        + md_table(pivot("median_abs_pct"), {c: f for c in cols})
        + "\n\n**Calibration of the variability models** (a calibrated model puts 10% of outcomes below its "
        "10th percentile, 80% inside the 10-90 band and 10% above; KS p-value tests uniform percentiles; median_cv = s.e. / ultimate, i.e. how wide the model is)\n\n"
        + md_table(cal_fmt, {"below_p10": f, "inside_p10_p90": f, "above_p90": f, "median_pct": f,
                             "ks_pvalue": lambda v: f"{v:.3f}", "median_cv": lambda v: f"{v:.1%}"})
        + "\n\n![calibration](figures/backtest_calibration.png)\n"
    ), df, acc, cal


def fig_mcl(df: pd.DataFrame, path: Path) -> None:
    series = ["CL Paid", "CL Incurred", "MCL Paid", "MCL Incurred"]
    lines = list(LINES)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, (m, c) in enumerate(zip(series, C)):
        err = (df[f"ult::{m}"] - df.actual_ult) / df.actual_unpaid
        med = [err[df.lob == l].median() for l in lines]
        y = np.arange(len(lines)) + (k - 1.5) * 0.12
        ax.scatter(med, y, color=c, s=46, label=m, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(np.arange(len(lines)), [LINES[l] for l in lines])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Median reserve error (% of actual unpaid)")
    ax.set_title("Munich chain ladder pulls paid and incurred together")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def extensions_section(df: pd.DataFrame, cal: pd.DataFrame) -> str:
    fig_mcl(df, FIG / "mcl_by_line.png")

    def rel(m):
        return (df[f"ult::{m}"] - df.actual_ult) / df.actual_unpaid

    gap_cl = (df["ult::CL Paid"] / df["ult::CL Incurred"] - 1).abs().median()
    gap_mcl = (df["ult::MCL Paid"] / df["ult::MCL Incurred"] - 1).abs().median()
    mcl_tbl = pd.DataFrame({m: {"median error": rel(m).median(), "median |error|": rel(m).abs().median(),
                                "within ±20%": (rel(m).abs() <= 0.2).mean()}
                            for m in ["CL Paid", "CL Incurred", "MCL Paid", "MCL Incurred"]}).T

    sh = cal.set_index(["model", "basis"]).loc[
        [("ODP bootstrap", "incurred"), ("ODP bootstrap (Shapland)", "incurred"),
         ("ODP bootstrap", "paid"), ("ODP bootstrap (Shapland)", "paid")],
        ["below_p10", "inside_p10_p90", "above_p90", "ks_pvalue", "median_cv"]]

    ts = bt.tail_study()
    ts.to_csv(OUT / "tail_study.csv", index=False)
    errs = ["err::exponential", "err::inverse_power"]
    tail_tbl = ts.groupby("lob")[errs].median().loc[list(LINES)]
    tail_tbl["abs_exp"] = ts.groupby("lob")["err::exponential"].apply(lambda v: v.abs().median()).loc[list(LINES)]
    tail_tbl.loc["all lines"] = [*ts[errs].median(), ts["err::exponential"].abs().median()]
    tail_tbl.columns = ["exponential: median error", "inverse power: median error", "exponential: median |error|"]

    it = bt.industry_tails()
    it["paid tail (inverse power)"] = bt.industry_tails(curve="inverse_power")["paid_tail_curve"]
    it = it.rename(columns={"paid_tail_curve": "paid tail (exponential)",
                            "incurred_over_paid_10": "case incurred / paid at age 10",
                            "bulk_ibnr_10": "IBNR still booked at age 10"})
    it.index = [LINES[l] for l in it.index]

    def f(v):
        return f"{v:.0%}"

    def fs(v):
        return f"{v:+.0%}"

    def f3(v):
        return f"{v:.3f}"

    return (
        "## 4. Extensions\n\n"
        "### 4.1 Munich chain ladder\n\n"
        "Quarg & Mack (2004) regress each year's next development factor on its current paid/incurred "
        "ratio residual: a year that is under-paid relative to its case incurred develops faster on paid "
        "and slower on incurred. Validated against chainladder-python to the cent (section 1).\n\n"
        + md_table(mcl_tbl, {"median error": fs, "median |error|": f, "within ±20%": f})
        + f"\n\nMedian gap between the paid and incurred ultimates: **{gap_cl:.1%}** with separate chain "
        f"ladders, **{gap_mcl:.1%}** with MCL.\n\n![mcl](figures/mcl_by_line.png)\n\n"
        "### 4.2 Berquist-Sherman without claim counts\n\n"
        "Schedule P in this database has no claim counts, so the classic disposal-rate adjustment is not "
        "available. The natural substitute restates historical paid to the latest diagonal's paid/incurred "
        "ratio (`reserving/adjustments.py`). On the data it reproduces incurred chain ladder *exactly*, and "
        "the algebra shows why: every adjusted column is a constant multiple of the incurred column, so the "
        "projection telescopes to `U_CL,incurred x r_10`, and the paid-to-incurred tail cancels `r_10`. "
        "A settlement adjustment that measures speed against incurred can only return the incurred answer; "
        "the independent information in Berquist-Sherman comes from the claim counts. (Asserted in "
        "`tests/test_extensions.py`.)\n\n"
        "### 4.3 Shapland refinements to the bootstrap\n\n"
        "Hat-matrix residual adjustment and separate scale parameters for groups of development ages "
        "(Shapland 2016). On incurred data, where tiny late-age incremental means inflated a single scale "
        "parameter, the refinements make the distribution materially narrower while keeping coverage close "
        "to nominal. On paid data they change little, because the paid failure is a biased mean, not a "
        "mis-sized spread.\n\n"
        + md_table(sh, {"below_p10": f, "inside_p10_p90": f, "above_p90": f,
                        "ks_pvalue": lambda v: f"{v:.3f}", "median_cv": lambda v: f"{v:.1%}"})
        + "\n\n### 4.4 Tail factors\n\n"
        "**Can a curve extrapolate development?** Each company's paid triangle is cut at age 5. Exponential "
        "and inverse-power curves are fitted to the factors for ages 2-5 and asked to predict the development "
        "from age 6 to 10, which the triangle actually shows. Error is relative to the true development "
        "portion (CDF - 1).\n\n"
        + md_table(tail_tbl, {tail_tbl.columns[0]: fs, tail_tbl.columns[1]: fs, tail_tbl.columns[2]: f})
        + "\n\nThe exponential curve is close to unbiased overall but misses by about half the development "
        "for a typical single company; the inverse power curve roughly doubles it. Tails belong on industry "
        "aggregates, and the choice of curve is itself a material assumption.\n\n"
        "**How far from ultimate is lag 10?** Industry aggregates of the clean companies:\n\n"
        + md_table(it[["case incurred / paid at age 10", "IBNR still booked at age 10",
                       "paid tail (exponential)", "paid tail (inverse power)"]],
                   {"case incurred / paid at age 10": f3, "IBNR still booked at age 10": lambda v: f"{v:.1%}",
                    "paid tail (exponential)": f3, "paid tail (inverse power)": f3})
        + "\n\nFor the auto lines lag 10 is effectively ultimate. Workers' comp still had about 6% of losses "
        "in case reserves and 3% in IBNR at age 10, so the backtest's 'actual' understates true ultimate "
        "there by a few percent, a shortfall every method shares.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000, help="bootstrap simulations per triangle in the backtest")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Reserving engine report\n\n"
        "Generated by `scripts/build_report.py` from the CAS Loss Reserving Database "
        "(NAIC Schedule P, accident years 1988-1997, valued 12/1997, outcomes through 2006). "
        "All amounts in $000s, net of reinsurance. See the [README](../README.md) for method notes and the "
        "interpretation of these results.\n",
        benchmark_section(),
        "## 2. Company deep dives\n",
    ]
    for i, (lob, code, headline) in enumerate(SHOWCASES, 1):
        parts.append(showcase_section(i, lob, code, headline, sims=10_000))
    text, df, acc, cal = backtest_section(args.sims)
    parts.append(text)
    parts.append(extensions_section(df, cal))
    (OUT / "REPORT.md").write_text("\n".join(parts))
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
