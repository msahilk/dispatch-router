"""Cost-quality evaluation of routers against recorded Haiku/Sonnet answers.

Inputs per test prompt i:
  h[i], s[i]    1 if Haiku / Sonnet answered correctly
  ch[i], cs[i]  recorded cost of the Haiku / Sonnet call
  score[i]      router's p(Haiku correct)

A router at threshold tau sends prompt i to Sonnet when score[i] < tau. Sweeping tau
is the same as sending the k lowest-scoring prompts to Sonnet for k = 0..n, so a
curve is evaluated at every k (every achievable Sonnet-call fraction k/n).

Metrics (RouteLLM, Ong et al. 2025):
  PGR(k)   = (acc(k) - acc_haiku) / (acc_sonnet - acc_haiku)
  APGR     = area under PGR vs. fraction-of-calls-to-Sonnet, over [0, 1]
  CPT(x)   = smallest fraction of Sonnet calls whose PGR >= x
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class Curve:
    frac_sonnet: np.ndarray  # k/n, k = 0..n
    accuracy: np.ndarray
    cost: np.ndarray  # absolute USD
    acc_haiku: float
    acc_sonnet: float
    cost_sonnet: float  # all-Sonnet cost (USD)

    @property
    def pgr(self) -> np.ndarray:
        gap = self.acc_sonnet - self.acc_haiku
        return (self.accuracy - self.acc_haiku) / gap if gap != 0 else np.full_like(self.accuracy, np.nan)

    @property
    def cost_pct(self) -> np.ndarray:
        return 100 * self.cost / self.cost_sonnet


def order_for(score: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Indices in the order prompts move to Sonnet (lowest score first). Ties are
    broken randomly when rng is given, else by position (stable)."""
    if rng is None:
        return np.argsort(score, kind="stable")
    return np.lexsort((rng.random(len(score)), score))


def curve(h, s, ch, cs, order: np.ndarray) -> Curve:
    h, s, ch, cs = (np.asarray(a, dtype=float) for a in (h, s, ch, cs))
    n = len(h)
    dacc = np.concatenate([[0.0], np.cumsum((s - h)[order])])
    dcost = np.concatenate([[0.0], np.cumsum((cs - ch)[order])])
    return Curve(
        frac_sonnet=np.arange(n + 1) / n,
        accuracy=(h.sum() + dacc) / n,
        cost=ch.sum() + dcost,
        acc_haiku=h.mean(),
        acc_sonnet=s.mean(),
        cost_sonnet=cs.sum(),
    )


def oracle_order(h, s) -> np.ndarray:
    """Haiku-wrong & Sonnet-right first, then Haiku-wrong & Sonnet-wrong, then Haiku-right."""
    h, s = np.asarray(h), np.asarray(s)
    return np.lexsort((-s, h))


def random_curve(h, s, ch, cs) -> Curve:
    """Expected curve of the random router: a fraction f of prompts, chosen uniformly,
    go to Sonnet. Accuracy and cost are exactly linear in f."""
    h, s, ch, cs = (np.asarray(a, dtype=float) for a in (h, s, ch, cs))
    n = len(h)
    f = np.arange(n + 1) / n
    return Curve(f, h.mean() + f * (s.mean() - h.mean()), ch.sum() + f * (cs.sum() - ch.sum()),
                 h.mean(), s.mean(), cs.sum())


def apgr(c: Curve) -> float:
    return float(_trapz(c.pgr, c.frac_sonnet))


def cpt(c: Curve, x: float) -> float:
    """Minimum fraction of calls to Sonnet with PGR >= x (NaN if never reached)."""
    hit = np.nonzero(c.pgr >= x - 1e-12)[0]
    return float(c.frac_sonnet[hit[0]]) if len(hit) else float("nan")


def cost_at_quality(c: Curve, rel: float) -> float:
    """Minimum cost (% of all-Sonnet) whose accuracy >= rel * Sonnet accuracy."""
    ok = np.nonzero(c.accuracy >= rel * c.acc_sonnet - 1e-12)[0]
    return float(c.cost_pct[ok].min()) if len(ok) else float("nan")


def at_frac(c: Curve, f: float, attr: str = "accuracy") -> float:
    k = int(round(f * (len(c.frac_sonnet) - 1)))
    return float(getattr(c, attr)[k])


def metrics(c: Curve) -> dict[str, float]:
    return {
        "acc_haiku": c.acc_haiku,
        "acc_sonnet": c.acc_sonnet,
        "apgr": apgr(c),
        "cpt50": cpt(c, 0.5),
        "cpt80": cpt(c, 0.8),
        "cost_pct_at_95": cost_at_quality(c, 0.95),
        "cost_pct_at_99": cost_at_quality(c, 0.99),
        "cost_pct_at_100": cost_at_quality(c, 1.0),
        "acc_at_30": at_frac(c, 0.30),
        "cost_pct_at_30": at_frac(c, 0.30, "cost_pct"),
        "max_accuracy": float(c.accuracy.max()),
    }


def router_metrics(h, s, ch, cs, score) -> dict[str, float]:
    m = metrics(curve(h, s, ch, cs, order_for(np.asarray(score))))
    h = np.asarray(h)
    m["auroc"] = roc_auc_score(h, score) if 0 < h.sum() < len(h) else float("nan")
    return m


def bootstrap(h, s, ch, cs, score, n_boot: int = 1000, seed: int = 0, fn=router_metrics) -> dict[str, tuple]:
    """95% percentile CIs: resample test prompts with replacement, keep scores fixed."""
    h, s, ch, cs, score = (np.asarray(a) for a in (h, s, ch, cs, score))
    rng = np.random.default_rng(seed)
    n = len(h)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        draws.append(fn(h[i], s[i], ch[i], cs[i], score[i]))
    out = {}
    for k in draws[0]:
        v = np.array([d[k] for d in draws], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return out


def paired_bootstrap_diff(h, s, ch, cs, score_a, score_b, key: str, n_boot: int = 1000, seed: int = 0):
    """CI for metric(a) - metric(b) on the same resampled prompts."""
    h, s, ch, cs, a, b = (np.asarray(x) for x in (h, s, ch, cs, score_a, score_b))
    rng = np.random.default_rng(seed)
    n, diffs = len(h), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(router_metrics(h[i], s[i], ch[i], cs[i], a[i])[key]
                     - router_metrics(h[i], s[i], ch[i], cs[i], b[i])[key])
    d = np.array(diffs)
    d = d[~np.isnan(d)]
    point = router_metrics(h, s, ch, cs, a)[key] - router_metrics(h, s, ch, cs, b)[key]
    return point, (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
