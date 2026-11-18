"""Glue between prompt items and the API cache: build specs, collect graded rows."""

from __future__ import annotations

import pandas as pd

from .config import load_config
from .llm import CallSpec, ResponseCache
from .prompts import grade


def specs_for(items: pd.DataFrame, model_key: str) -> list[CallSpec]:
    mt = load_config()["max_tokens"]
    return [CallSpec(model_key, r.prompt, mt[r.dataset], r.dataset) for r in items.itertuples()]


def graded(items: pd.DataFrame, model_key: str, cache: ResponseCache | None = None) -> pd.DataFrame:
    """One row per item with the recorded response and its grade. Items with no
    cached response yet are skipped."""
    cache = cache or ResponseCache()
    rows = []
    for r, spec in zip(items.itertuples(), specs_for(items, model_key)):
        rec = cache.get(spec)
        if rec is None:
            continue
        g = grade(r.answer_type, r.gold, rec["text"])
        rows.append(
            {
                "id": r.id,
                "dataset": r.dataset,
                "category": r.category,
                "difficulty": r.difficulty,
                "split": getattr(r, "split", ""),
                "model_id": rec["model_id"],
                "input_tokens": rec["input_tokens"],
                "output_tokens": rec["output_tokens"],
                "cost_usd": rec["cost_usd"],
                "latency_s": rec["latency_s"],
                "mode": rec["mode"],
                "stop_reason": rec["stop_reason"],
                "gold": r.gold,
                "extracted": g["extracted"],
                "correct": bool(g["correct"]),
                "parse_fail": bool(g["parse_fail"]),
                "raw_output": rec["text"],
            }
        )
    return pd.DataFrame(rows)


def accuracy_report(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (
        df.groupby(by)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"), parse_fail_rate=("parse_fail", "mean"),
             mean_out_tokens=("output_tokens", "mean"), cost_usd=("cost_usd", "sum"))
        .round(4)
    )
