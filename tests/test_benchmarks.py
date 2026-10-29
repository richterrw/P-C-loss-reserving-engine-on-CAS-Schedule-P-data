"""Validate against published results for the classic RAA and Taylor-Ashe triangles.

Reference figures are those printed by R ``ChainLadder::MackChainLadder`` and
England & Verrall (2002) for the same data.
"""

import numpy as np
import pytest

from reserving.benchmarks import load_benchmark
from reserving.bf import bornhuetter_ferguson, benktander, cape_cod
from reserving.bootstrap import odp_bootstrap
from reserving.chainladder import chain_ladder
from reserving.mack import mack


@pytest.fixture(scope="module")
def raa():
    return load_benchmark("raa")


@pytest.fixture(scope="module")
def genins():
    return load_benchmark("genins")


def test_raa_chain_ladder(raa):
    cl = chain_ladder(raa)
    assert raa.latest.sum() == pytest.approx(160_987)
    assert cl.reserve.sum() == pytest.approx(52_135, abs=1)
    assert cl.ultimate[-1] == pytest.approx(18_402, abs=1)


def test_raa_mack(raa):
    m = mack(raa)
    expected_se = [0, 206, 623, 747, 1_469, 2_002, 2_209, 5_358, 6_333, 24_566]
    np.testing.assert_allclose(m.se, expected_se, atol=1)
    assert m.total_se == pytest.approx(26_909, abs=1)


def test_genins_mack(genins):
    m = mack(genins)
    assert m.reserve.sum() == pytest.approx(18_680_856, abs=1)
    assert m.total_se == pytest.approx(2_447_095, abs=1)


def test_genins_bootstrap(genins):
    b = odp_bootstrap(genins, n_sims=10_000, seed=1)
    det = chain_ladder(genins).reserve.sum()
    # Bootstrap mean is close to (slightly above) the chain ladder reserve.
    assert b.total.mean() == pytest.approx(det, rel=0.03)
    # England & Verrall (2002) report a bootstrap prediction error of about 16% of reserves.
    assert 0.14 < b.total_se / det < 0.18
    # Without process risk the spread is parameter error only, and smaller.
    b0 = odp_bootstrap(genins, n_sims=4_000, seed=1, process=False)
    assert b0.total_se < b.total_se


def test_bf_identity(raa):
    cl = chain_ladder(raa)
    prem = np.full(raa.n, 30_000.0)
    bf = bornhuetter_ferguson(cl, prem, 0.7)
    p = 1 / cl.cdf
    np.testing.assert_allclose(cl.ultimate - bf.ultimate, (1 - p) * (cl.ultimate - bf.prior_ultimate))
    np.testing.assert_allclose(bf.credibility[1:], p[1:])
    bk = benktander(cl, prem, 0.7)
    np.testing.assert_allclose(bk.credibility[1:], 1 - (1 - p[1:]) ** 2)


def test_cape_cod_reproduces_reported(raa):
    cl = chain_ladder(raa)
    prem = np.linspace(20_000, 40_000, raa.n)
    cc = cape_cod(cl, prem)
    # Cape Cod ELR is chosen so that expected reported to date equals actual reported.
    assert (cc.elr[0] * prem / cl.cdf).sum() == pytest.approx(raa.latest.sum())
