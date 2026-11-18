"""Load and sample GSM8K / MMLU into prompt items, and build train/val splits.

An item is a dict:
  id, dataset, category (MMLU subject / backend category), difficulty,
  prompt, gold, answer_type, split
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ROOT, load_config
from .prompts import LETTERS, gsm8k_prompt, mmlu_prompt

RAW = ROOT / "data" / "raw"


def _load(name: str, config: str):
    from datasets import load_dataset

    return load_dataset(name, config, cache_dir=str(RAW))


def gsm8k_gold(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


def gsm8k_items(split: str) -> pd.DataFrame:
    ds = _load("openai/gsm8k", "main")[split]
    rows = [
        {
            "id": f"gsm8k-{split}-{i}",
            "dataset": "gsm8k",
            "category": "math_word_problem",
            "difficulty": "",
            "question": r["question"],
            "prompt": gsm8k_prompt(r["question"]),
            "gold": gsm8k_gold(r["answer"]),
            "answer_type": "number",
        }
        for i, r in enumerate(ds)
    ]
    return pd.DataFrame(rows)


def mmlu_items(splits: list[str]) -> pd.DataFrame:
    dd = _load("cais/mmlu", "all")
    rows = []
    for split in splits:
        for i, r in enumerate(dd[split]):
            rows.append(
                {
                    "id": f"mmlu-{split}-{i}",
                    "dataset": "mmlu",
                    "category": r["subject"],
                    "difficulty": "",
                    "question": r["question"],
                    "prompt": mmlu_prompt(r["question"], r["subject"], r["choices"]),
                    "gold": LETTERS[r["answer"]],
                    "answer_type": "letter",
                }
            )
    return pd.DataFrame(rows)


def stratified_sample(df: pd.DataFrame, n: int, by: str, seed: int) -> pd.DataFrame:
    """Sample n rows keeping each group's share (largest-remainder rounding)."""
    if n >= len(df):
        return df.sample(frac=1, random_state=seed)
    shares = df[by].value_counts() * n / len(df)
    alloc = shares.astype(int)
    leftover = n - alloc.sum()
    for g in (shares - alloc).sort_values(ascending=False).index[:leftover]:
        alloc[g] += 1
    parts = [df[df[by] == g].sample(k, random_state=seed) for g, k in alloc.items() if k > 0]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_train_pool() -> pd.DataFrame:
    cfg = load_config()["sampling"]
    seed = cfg["seed"]
    g = gsm8k_items("train").sample(cfg["train"]["gsm8k"], random_state=seed)
    m = stratified_sample(mmlu_items(["validation", "dev"]), cfg["train"]["mmlu"], by="category", seed=seed)
    pool = pd.concat([g, m]).reset_index(drop=True)
    tr, va = train_test_split(
        pool.index, test_size=cfg["val_fraction"], random_state=seed, stratify=pool["dataset"]
    )
    pool["split"] = "train"
    pool.loc[va, "split"] = "val"
    return pool


def normalize_text(s: str) -> str:
    """For overlap checks: lowercase, collapse whitespace, drop punctuation."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def write_jsonl(df: pd.DataFrame | list[dict[str, Any]], file: Path) -> None:
    rows = df.to_dict("records") if isinstance(df, pd.DataFrame) else df
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(file: Path) -> pd.DataFrame:
    with open(file) as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])
