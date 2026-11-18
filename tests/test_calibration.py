import numpy as np
import pytest

from dispatch_router.calibrate import Platt, brier, ece, realized_rate, threshold_for_rate


def test_ece_zero_when_perfectly_calibrated():
    p = np.repeat([0.15, 0.85], 100)
    y = np.concatenate([np.r_[np.ones(15), np.zeros(85)], np.r_[np.ones(85), np.zeros(15)]])
    assert ece(p, y) == pytest.approx(0.0)
    assert ece(np.full(200, 0.95), y) == pytest.approx(0.45)


def test_brier():
    assert brier([1, 0], [1, 0]) == 0 and brier([0.5], [1]) == 0.25


def test_platt_fixes_overconfident_scores():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.6, 1.0, 5000)
    y = (rng.random(5000) < true_p).astype(int)
    raw = true_p ** 0.2  # squashed towards 1 -> overconfident
    cal = Platt().fit(raw, y).transform(raw)
    assert ece(cal, y) < ece(raw, y) / 3


def test_threshold_controls_rate():
    s = np.linspace(0, 1, 1001)
    tau = threshold_for_rate(s, 0.2)
    assert realized_rate(s, tau) == pytest.approx(0.2, abs=0.002)
