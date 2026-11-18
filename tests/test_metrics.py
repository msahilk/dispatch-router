import numpy as np
import pytest

from dispatch_router.evaluate import (
    apgr, cost_at_quality, cpt, curve, oracle_order, order_for, random_curve, router_metrics,
)

# Toy test set: 4 prompts. Haiku right on 0,1; Sonnet right on 0,1,2,3.
H = np.array([1, 1, 0, 0])
S = np.array([1, 1, 1, 1])
CH = np.full(4, 1.0)
CS = np.full(4, 10.0)


def test_curve_endpoints():
    c = curve(H, S, CH, CS, order_for(np.array([0.9, 0.8, 0.1, 0.2])))
    assert c.accuracy[0] == 0.5 and c.accuracy[-1] == 1.0  # all Haiku / all Sonnet
    assert c.cost_pct[0] == pytest.approx(10.0) and c.cost_pct[-1] == pytest.approx(100.0)


def test_perfect_router_pgr_and_metrics():
    # prompts 2,3 (Haiku wrong) have the lowest scores -> moved to Sonnet first
    c = curve(H, S, CH, CS, order_for(np.array([0.9, 0.8, 0.1, 0.2])))
    assert list(c.pgr) == [0, 0.5, 1, 1, 1]
    assert cpt(c, 0.5) == 0.25 and cpt(c, 0.8) == 0.5
    # area under [0,.5,1,1,1] on x=[0,.25,.5,.75,1] = 0.0625+0.1875+0.25+0.25
    assert apgr(c) == pytest.approx(0.75)
    assert cost_at_quality(c, 1.0) == pytest.approx(100 * (2 + 20) / 40)


def test_worst_router():
    c = curve(H, S, CH, CS, order_for(np.array([0.1, 0.2, 0.9, 0.8])))
    assert list(c.pgr) == [0, 0, 0, 0.5, 1]
    assert cpt(c, 0.8) == 1.0


def test_random_curve_is_diagonal():
    c = random_curve(H, S, CH, CS)
    assert np.allclose(c.pgr, c.frac_sonnet)
    assert apgr(c) == pytest.approx(0.5)
    assert cpt(c, 0.5) == 0.5


def test_random_router_matches_expected_on_average():
    rng = np.random.default_rng(0)
    h = rng.integers(0, 2, 200)
    s = np.maximum(h, rng.integers(0, 2, 200))
    ch, cs = np.full(200, 1.0), np.full(200, 20.0)
    avg = np.mean([apgr(curve(h, s, ch, cs, rng.permutation(200))) for _ in range(300)])
    assert avg == pytest.approx(0.5, abs=0.02)


def test_oracle_is_upper_envelope():
    rng = np.random.default_rng(1)
    h, s = rng.integers(0, 2, 100), rng.integers(0, 2, 100)
    ch, cs = np.ones(100), np.full(100, 5.0)
    o = curve(h, s, ch, cs, oracle_order(h, s))
    for _ in range(20):
        r = curve(h, s, ch, cs, order_for(rng.random(100)))
        assert np.all(o.accuracy >= r.accuracy - 1e-12)


def test_router_metrics_has_auroc():
    m = router_metrics(H, S, CH, CS, np.array([0.9, 0.8, 0.1, 0.2]))
    assert m["auroc"] == 1.0 and m["apgr"] == pytest.approx(0.75)
