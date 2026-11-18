"""Logistic regression on prompt embeddings (the main router)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class LogRegRouter:
    name = "logreg"

    def __init__(self, C_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0), seed: int = 42):
        self.C_grid = C_grid
        self.seed = seed
        self.tuning: dict[str, float] = {}

    def _model(self, C, cw):
        return LogisticRegression(C=C, class_weight=cw, max_iter=5000, random_state=self.seed)

    def fit(self, X_tr, y_tr, X_val, y_val, **_):
        best = None
        for cw in (None, "balanced"):
            for C in self.C_grid:
                m = self._model(C, cw).fit(X_tr, y_tr)
                auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
                self.tuning[f"C={C},class_weight={cw}"] = auc
                if best is None or auc > best[0]:
                    best = (auc, C, cw, m)
        _, self.C, self.class_weight, self.model = best
        return self

    def score(self, X, is_train: bool = False, **_) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def describe(self) -> dict:
        return {"C": self.C, "class_weight": self.class_weight,
                "val_auroc_grid": {k: round(v, 4) for k, v in self.tuning.items()}}
