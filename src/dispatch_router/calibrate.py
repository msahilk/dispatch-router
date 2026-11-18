"""Probability calibration and threshold calibration.

We fit Platt scaling (a 1-D logistic regression on the logit of the raw score) on
the validation split only. Isotonic regression is the other standard choice, but
it is a step function with one free parameter per step; with only ~46 "Haiku
wrong" examples in validation it would overfit, so Platt's 2 parameters it is.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, score, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score)[:, None], np.asarray(y).astype(int))
        return self

    def transform(self, score) -> np.ndarray:
        return self.lr.predict_proba(_logit(score)[:, None])[:, 1]

    @property
    def params(self) -> dict:
        return {"a": float(self.lr.coef_[0, 0]), "b": float(self.lr.intercept_[0])}


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def reliability(p, y, n_bins: int = 10):
    """Equal-width bins on [0, 1]. Returns (bin_mean_pred, bin_frac_pos, bin_count)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    mp, fp, cnt = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, int)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b]:
            mp[b], fp[b] = p[m].mean(), y[m].mean()
    return mp, fp, cnt


def ece(p, y, n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted mean |predicted - actual| over bins."""
    mp, fp, cnt = reliability(p, y, n_bins)
    m = cnt > 0
    return float(np.sum(cnt[m] * np.abs(mp[m] - fp[m])) / cnt.sum())


def threshold_for_rate(val_score, target_sonnet_rate: float) -> float:
    """tau such that a fraction `target_sonnet_rate` of validation prompts score below it."""
    return float(np.quantile(np.asarray(val_score), target_sonnet_rate))


def realized_rate(score, tau: float) -> float:
    return float(np.mean(np.asarray(score) < tau))
