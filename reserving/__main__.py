"""Command line: run every method on one company.

    python -m reserving --lob othliab --company 1767
    python -m reserving --lob wkcomp --list
"""

from __future__ import annotations

import argparse

import pandas as pd

from .data import LINES, clean_companies, load_line
from .diagnostics import calendar_year_test, cl_bf_gap, correlation_test
from .engine import BASES, analyze


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m reserving", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lob", choices=list(LINES), required=True)
    ap.add_argument("--company", type=int, help="NAIC group code (GRCODE)")
    ap.add_argument("--list", action="store_true", help="list clean companies in the line, largest first")
    ap.add_argument("--sims", type=int, default=5000)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.2f}".format)

    if args.list or args.company is None:
        df = load_line(args.lob)
        codes = clean_companies(args.lob)
        prem = (df[(df.DevelopmentLag == 1) & df.GRCODE.isin(codes)]
                .groupby(["GRCODE", "GRNAME"]).EarnedPremNet.sum().sort_values(ascending=False))
        print(prem.rename("net EP 1988-97 ($000)").to_string())
        return

    a = analyze(args.lob, args.company, n_sims=args.sims)
    d = a.data
    print(f"\n{d.name} | {LINES[d.lob]} | GRCODE {d.grcode}\n")
    print("Point estimates (target: case incurred at lag 10)")
    print(a.summary().to_string())
    print("\nBy accident year")
    cols = ["premium", "paid", "case_incurred", "prior_elr", "CL Paid", "CL Incurred", "BF Incurred",
            "Cape Cod Incurred", "Company booked", "Actual (lag 10)"]
    print(a.by_origin()[cols].to_string())
    cl = a.cl["paid"]
    print("\nPaid CL - BF decomposition: (1 - 1/CDF) x (CL ult - prior ult)")
    print(cl_bf_gap(cl.ultimate, a.prior_elr * d.premium, cl.cdf, d.origins).to_string())
    print("\nVariability (own basis, to lag 10)")
    for b in BASES:
        m, bs = a.mack[b], a.boot[b]
        latest = d.triangle(b).latest.sum()
        actual = d.actual_ultimate(b).sum() - latest
        print(f"  {b:9s} CL {m.reserve.sum():>14,.0f}  Mack se {m.total_se:>12,.0f}  "
              f"boot mean {bs.total.mean():>14,.0f}  boot se {bs.total_se:>12,.0f}  "
              f"actual {actual:>14,.0f}  (boot pct {bs.rank_of(actual):.0%})")
        print(f"            calendar-year test {calendar_year_test(d.triangle(b))}  correlation test {correlation_test(d.triangle(b))}")


if __name__ == "__main__":
    main()
