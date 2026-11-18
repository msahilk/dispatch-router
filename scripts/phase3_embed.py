"""Embed every train/validation/test prompt (free, local). Cached; re-runs skip work.

    uv run python scripts/phase3_embed.py
"""

import json

from dispatch_router.config import load_config, path
from dispatch_router.datasets import read_jsonl
from dispatch_router.embed import device, embed_texts, save

SETS = {"train_pool": "train_pool.jsonl", "test_public": "test_public.jsonl", "backend": "backend_set.jsonl"}


def main() -> None:
    print(f"model={load_config()['embeddings']['model']} device={device()}")
    for name, file in SETS.items():
        df = read_jsonl(path("prompts_dir") / file)
        ids_file = path("embeddings_dir") / f"{name}.ids.json"
        if ids_file.exists() and json.loads(ids_file.read_text()) == list(df["id"]):
            print(f"{name}: cached ({len(df)})")
            continue
        X = embed_texts(df["prompt"].tolist())
        save(name, df, X)
        print(f"{name}: {X.shape} saved")


if __name__ == "__main__":
    main()
