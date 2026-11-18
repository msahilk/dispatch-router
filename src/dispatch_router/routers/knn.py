"""k-nearest-neighbours router: score = similarity-weighted share of the k most
similar training prompts that Haiku got right."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


class KNNRouter:
    name = "knn"

    def __init__(self, k_grid=(5, 10, 20, 30, 50, 100)):
        self.k_grid = k_grid
        self.k = None
        self.tuning: dict[int, float] = {}

    def _score(self, X: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
        sims = X @ self.X.T  # embeddings are L2-normalized -> cosine similarity
        if exclude_self:
            np.fill_diagonal(sims, -np.inf)
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        s = np.take_along_axis(sims, idx, axis=1)
        w = np.exp((s - s.max(axis=1, keepdims=True)) / 0.05)  # softmax-style distance weighting
        return (w * self.y[idx]).sum(1) / w.sum(1)

    def fit(self, X_tr, y_tr, X_val, y_val, **_):
        self.X, self.y = X_tr, y_tr.astype(float)
        for k in self.k_grid:
            self.tuning[k] = roc_auc_score(y_val, self._score(X_val, k))
        self.k = max(self.tuning, key=self.tuning.get)
        return self

    def score(self, X, is_train: bool = False, **_) -> np.ndarray:
        # On training prompts, leave each prompt out of its own neighbourhood.
        return self._score(X, self.k, exclude_self=is_train)

    def describe(self) -> dict:
        return {"k": self.k, "val_auroc_by_k": {str(k): round(v, 4) for k, v in self.tuning.items()}}
