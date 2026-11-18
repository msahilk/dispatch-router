"""Build the backend-engineering prompt set. Free: no API calls.

Every gold answer is produced by executing the Python snippet, running the SQL in
sqlite3, or evaluating the formula. Prints a summary table and random samples.

    uv run python scripts/phase2_build_backend_set.py
"""

import random

import pandas as pd

from dispatch_router.backend_set import build_backend_set
from dispatch_router.config import load_config, path
from dispatch_router.datasets import write_jsonl


def main() -> None:
    items, dropped = build_backend_set(load_config()["sampling"]["seed"])
    write_jsonl(items, path("prompts_dir") / "backend_set.jsonl")
    df = pd.DataFrame(items)
    table = pd.crosstab(df.category, df.difficulty, margins=True, margins_name="total")
    print(f"Backend set: {len(df)} items -> data/prompts/backend_set.jsonl")
    print(f"Dropped during verification: {dropped}\n")
    print(table.to_string(), "\n")
    print(f"easy share: {(df.difficulty == 'easy').mean():.0%}\n")
    for it in random.Random(0).sample(items, 15):
        q = it["question"].replace("\n", "\n      ")
        print(f"[{it['id']}] ({it['difficulty']}, {it['answer_type']})\n      {q}\n   -> GOLD: {it['gold']}\n")


if __name__ == "__main__":
    main()
