"""Assemble per-prompt evaluation frames: labels, costs and router scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ROOT, path
from .datasets import read_jsonl

RAW_SCORES = ROOT / "data" / "scores" / "raw_scores.csv"
CAL_SCORES = ROOT / "data" / "scores" / "calibrated_scores.csv"
LEARNED = ["prior", "knn", "logreg", "distilbert"]


def train_pool() -> pd.DataFrame:
    pool = read_jsonl(path("prompts_dir") / "train_pool.jsonl")
    lab = read_jsonl(path("responses_dir") / "haiku_train.jsonl")[["id", "correct"]]
    return pool.merge(lab, on="id")


def prior_scores(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Baseline 'router' with no language understanding: Haiku's training accuracy
    for the prompt's MMLU subject, else its dataset, else overall."""
    by_cat = train.groupby("category").correct.mean()
    by_ds = train.groupby("dataset").correct.mean()
    overall = train.correct.mean()
    return np.array([by_cat.get(c, by_ds.get(d, overall)) for c, d in zip(df.category, df.dataset)], dtype=float)


def test_frame() -> pd.DataFrame:
    """One row per test prompt with both models' correctness and cost."""
    r = path("responses_dir")
    h = read_jsonl(r / "test_haiku.jsonl")
    s = read_jsonl(r / "test_sonnet.jsonl")
    cols = ["id", "correct", "cost_usd", "latency_s", "parse_fail", "output_tokens"]
    m = h[["id", "dataset", "category", "difficulty"] + cols[1:]].merge(s[cols], on="id", suffixes=("_h", "_s"))
    backend = read_jsonl(path("prompts_dir") / "backend_set.jsonl")[["id", "tier"]]
    m = m.merge(backend, on="id", how="left")
    m["tier"] = m["tier"].fillna("")
    m["set"] = np.where(m.dataset == "backend", "backend", "test_public")
    return m


def subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pub, be = df[df.set == "test_public"], df[df.set == "backend"]
    return {
        "public": pub,
        "gsm8k": pub[pub.dataset == "gsm8k"],
        "mmlu": pub[pub.dataset == "mmlu"],
        "backend": be,
        "backend_stress": be[be.tier == "stress"],
        "combined": df,
    }


def tie_broken(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """Add a tiny seeded jitter so routers with tied scores (e.g. the prior, which
    gives every backend prompt the same score) are split randomly instead of by
    file order. Never changes a strict ordering."""
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    gaps = np.diff(np.unique(score))
    eps = (gaps.min() if len(gaps) else 1.0) * 1e-3
    return score + eps * rng.random(len(score))


def with_router_scores(df: pd.DataFrame, cal: pd.DataFrame, routers: list[str]) -> pd.DataFrame:
    df = df.copy()
    for r in routers:
        df[r] = tie_broken(df.id.map(cal[cal.router == r].set_index("id").score_cal).values)
    assert not df[routers].isna().any().any(), "missing router scores for some test prompts"
    return df
