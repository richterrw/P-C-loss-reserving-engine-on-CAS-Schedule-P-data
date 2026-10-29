import numpy as np
import pytest

from reserving.benchmarks import load_benchmark
from reserving.chainladder import chain_ladder
from reserving.data import clean_companies, company
from reserving.diagnostics import (actual_vs_expected_diagonal, calendar_year_test, cl_bf_gap,
                                   correlation_test)
from reserving.engine import analyze, industry_elr, paid_tail
from reserving.triangle import Triangle


def test_company_squares_are_consistent():
    cd = company("wkcomp", 86)
    assert cd.paid_square.shape == (10, 10)
    assert cd.origins == tuple(range(1988, 1998))
    # Upper triangle masks exactly the post-1997 cells.
    assert np.isnan(cd.paid.values).sum() == 45
    # Case incurred = IncurLoss - BulkLoss, first row from the source file.
    assert cd.incurred_square[0, 0] == 367404 - 127737
    assert cd.paid_square[0, 0] == 70571


def test_clean_screen():
    codes = clean_companies("wkcomp")
    assert 86 in codes
    assert 38997 not in codes  # post-1997 data is frozen for this group


def test_industry_prior_is_reasonable():
    elr = industry_elr("ppauto")
    assert elr.shape == (10,)
    assert np.all((elr > 0.5) & (elr < 1.0))


def test_engine_point_estimates_share_target():
    a = analyze("othliab", 1767, variability=False)
    s = a.summary()
    actual = s.loc["Actual (lag 10)", "ultimate"]
    assert actual == pytest.approx(a.data.incurred_square[:, -1].sum())
    # Methods differ, but every estimate is in the right ballpark.
    assert ((s["ultimate"] / actual).between(0.8, 1.3)).all()
    assert paid_tail(a.data) >= 1.0


def test_gap_identity_on_company():
    a = analyze("medmal", 669, variability=False)
    cl = a.cl["paid"]
    gap = cl_bf_gap(cl.ultimate, a.prior_elr * a.data.premium, cl.cdf, a.data.origins)
    np.testing.assert_allclose(gap["cl_minus_bf"], a.estimates["CL Paid"] - a.estimates["BF Paid"])


def test_mack_assumption_tests_raa():
    raa = load_benchmark("raa")
    cy = calendar_year_test(raa)
    assert cy["Z"] == 14 and cy["E[Z]"] == pytest.approx(12.875)
    assert not cy["calendar_effect"]
    assert correlation_test(raa)["T"] == pytest.approx(0.0696, abs=1e-3)


def test_actual_vs_expected_flat_triangle():
    # A triangle that develops with constant factors has no surprise on the diagonal.
    f = np.array([2.0, 1.5, 1.2, 1.1, 1.05])
    sq = np.outer(np.arange(1, 7) * 100.0, np.concatenate([[1.0], np.cumprod(f)]))
    tri = Triangle.from_square(sq, range(2000, 2006))
    ave = actual_vs_expected_diagonal(tri)
    np.testing.assert_allclose(ave["a_over_e"], 1.0)
    np.testing.assert_allclose(chain_ladder(tri).ultimate, sq[:, -1])
