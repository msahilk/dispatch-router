"""Train the learned routers on Haiku's training labels and score every prompt.

Writes raw (uncalibrated) scores for train / val / test_public / backend to
data/scores/raw_scores.csv, plus tuning details to results/tables/.

    uv run python scripts/phase4_train_routers.py                     # all routers
    uv run python scripts/phase4_train_routers.py --routers knn logreg
"""

import argparse
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from dispatch_router import embed
from dispatch_router.config import ROOT, path
from dispatch_router.datasets import read_jsonl
from dispatch_router.routers import get

SCORES = ROOT / "data" / "scores" / "raw_scores.csv"
MODEL_DIR = ROOT / "models" / "distilbert"


def load_sets():
    p, r = path("prompts_dir"), path("responses_dir")
    pool = read_jsonl(p / "train_pool.jsonl")
    labels = read_jsonl(r / "haiku_train.jsonl")[["id", "correct"]]
    pool = pool.merge(labels, on="id", how="inner")
    X_pool = embed.load("train_pool", pool)
    tr, va = (pool.split == "train").values, (pool.split == "val").values
    sets = {
        "train": (pool[tr].reset_index(drop=True), X_pool[tr]),
        "val": (pool[va].reset_index(drop=True), X_pool[va]),
    }
    for name, file in (("test_public", "test_public.jsonl"), ("backend", "backend_set.jsonl")):
        df = read_jsonl(p / file)
        sets[name] = (df, embed.load(name, df))
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", nargs="+", default=["knn", "logreg", "distilbert"])
    ap.add_argument("--rescore-distilbert", action="store_true",
                    help="score all sets with the saved DistilBERT checkpoint, no training")
    args = ap.parse_args()

    sets = load_sets()
    (tr, Xtr), (va, Xva) = sets["train"], sets["val"]
    ytr, yva = tr.correct.astype(int).values, va.correct.astype(int).values
    print(f"train n={len(tr)} (Haiku wrong: {(ytr == 0).sum()})   val n={len(va)} (Haiku wrong: {(yva == 0).sum()})")

    old = pd.read_csv(SCORES) if SCORES.exists() else pd.DataFrame(columns=["id", "set", "router", "score"])
    old = old[~old.router.isin(args.routers)]
    rows, tables = [old], path("results_dir") / "tables"
    tables.mkdir(exist_ok=True)
    for name in args.routers:
        print(f"\n== {name}")
        if name == "distilbert" and args.rescore_distilbert:
            from dispatch_router.routers.distilbert import DistilBertRouter

            r = DistilBertRouter.load(MODEL_DIR)
            r.describe = lambda: json.loads((tables / "router_distilbert.json").read_text())
        else:
            r = get(name)
            r.fit(Xtr, ytr, Xva, yva, texts_tr=tr.prompt, texts_val=va.prompt)
            if name == "distilbert":
                r.save(MODEL_DIR)
        for set_name, (df, X) in sets.items():
            s = r.score(X, texts=df.prompt.tolist(), is_train=(set_name == "train"))
            rows.append(pd.DataFrame({"id": df.id, "set": set_name, "router": name, "score": s}))
            if "correct" in df:
                print(f"  {set_name:<5} AUROC = {roc_auc_score(df.correct.astype(int), s):.4f}")
        info = r.describe()
        (tables / f"router_{name}.json").write_text(json.dumps(info, indent=2, default=str))
        if name == "distilbert":
            pd.DataFrame(info["history"]).to_csv(tables / "distilbert_history.csv", index=False)
        print("  chosen:", {k: v for k, v in info.items() if k not in ("history", "val_auroc_grid", "val_auroc_by_k")})

    SCORES.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(SCORES, index=False)
    print(f"\nscores -> {SCORES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
