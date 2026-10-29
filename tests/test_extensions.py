"""Munich chain ladder, Berquist-Sherman proxy, Shapland bootstrap options, tails."""

import numpy as np
import pandas as pd
import pytest

from reserving.adjustments import settlement_adjusted_paid
from reserving.benchmarks import BENCH_DIR, load_benchmark
from reserving.bootstrap import auto_hetero_groups, fitted_incrementals, hat_diagonal, odp_bootstrap
from reserving.chainladder import chain_ladder
from reserving.data import company
from reserving.engine import paid_tail
from reserving.mack import mack_sigma2
from reserving.munich import munich_chain_ladder
from reserving.tail import curve_extrapolation_test, tail_factor
from reserving.triangle import Triangle, upper_mask


@pytest.fixture(scope="module")
def mcl():
    """Quarg & Mack (2004) paid/incurred example (R ChainLadder ``MCLpaid``/``MCLincurred``)."""
    d = pd.read_csv(BENCH_DIR / "mcl.csv")
    d["lag"] = d.development - d.origin + 1
    return Triangle.from_long(d, "origin", "lag", "paid"), Triangle.from_long(d, "origin", "lag", "incurred")


def test_munich_matches_reference(mcl):
    # Reference: chainladder-python 0.8 MunichAdjustment on the same data.
    r = munich_chain_ladder(*mcl)
    assert r.lambda_paid == pytest.approx(0.636021, abs=1e-5)
    assert r.lambda_incurred == pytest.approx(0.436187, abs=1e-5)
    assert r.paid_ultimate.sum() == pytest.approx(32_443.53, abs=0.01)
    assert r.incurred_ultimate.sum() == pytest.approx(32_707.82, abs=0.01)
    np.testing.assert_allclose(np.round(r.paid_ultimate), [2131, 2382, 4610, 6134, 4954, 4672, 7561])


def test_munich_narrows_paid_incurred_gap(mcl):
    paid, inc = mcl
    r = munich_chain_ladder(paid, inc)
    gap_cl = abs(chain_ladder(paid).ultimate.sum() / chain_ladder(inc).ultimate.sum() - 1)
    gap_mcl = abs(r.paid_ultimate.sum() / r.incurred_ultimate.sum() - 1)
    assert gap_mcl < gap_cl / 3


def test_log_linear_sigma(mcl):
    paid, _ = mcl
    from reserving.chainladder import development_factors
    s = np.sqrt(mack_sigma2(paid, development_factors(paid), "log-linear"))
    assert s[-1] == pytest.approx(0.0682, abs=1e-4)


def test_berquist_sherman_proxy_equals_incurred_cl():
    cd = company("othliab", 1767)
    adj = settlement_adjusted_paid(cd.paid, cd.incurred)
    np.testing.assert_allclose(adj.latest, cd.paid.latest)
    u = chain_ladder(adj, tail=paid_tail(cd)).ultimate
    np.testing.assert_allclose(u, chain_ladder(cd.incurred).ultimate, rtol=1e-12)


def test_hat_matrix_trace_equals_parameter_count():
    g = load_benchmark("genins")
    h = hat_diagonal(fitted_incrementals(g), upper_mask(g.n))
    assert np.nansum(h) == pytest.approx(2 * g.n - 1)
    assert h[0, -1] == pytest.approx(1.0) and h[-1, 0] == pytest.approx(1.0)


def test_hetero_groups_cover_all_ages():
    groups = auto_hetero_groups(10)
    assert sorted(a for g in groups for a in g) == list(range(10))


def test_shapland_options_tame_negative_development():
    # A wkcomp incurred triangle with near-zero late development blew up the plain bootstrap.
    tri = company("wkcomp", 3240).incurred
    plain = odp_bootstrap(tri, 2000, seed=3)
    shap = odp_bootstrap(tri, 2000, seed=3, hat_adjust=True, hetero=True)
    assert shap.total_se < plain.total_se / 2


def test_tail_curves():
    f = 1 + 0.5 * np.exp(-0.6 * np.arange(9))
    assert tail_factor(f, "exponential", start=0) == pytest.approx(np.prod(1 + 0.5 * np.exp(-0.6 * np.arange(9, 109))))
    sq = np.outer(np.full(10, 100.0), np.concatenate([[1.0], np.cumprod(f)]))
    t = curve_extrapolation_test(Triangle.from_square(sq, range(10)), k=5, start=0)
    assert t["exponential"] == pytest.approx(t["actual"])
