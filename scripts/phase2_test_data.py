"""Build the public test set and record BOTH models on it plus the backend set.

    uv run python scripts/phase2_test_data.py --estimate   # builds sets, prints cost, no calls
    uv run python scripts/phase2_test_data.py --run        # one Message Batch for both models
"""

import argparse

import pandas as pd

from dispatch_router.config import load_config, path
from dispatch_router.datasets import assert_no_overlap, build_public_test, read_jsonl, write_jsonl
from dispatch_router.llm import ResponseCache, check_budget, estimate, format_estimate, run_batch
from dispatch_router.runs import accuracy_report, graded, specs_for

PHASE = "test_both_models"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()["sampling"]
    pdir = path("prompts_dir")
    train = read_jsonl(pdir / "train_pool.jsonl")
    public = build_public_test(train, cfg["test"]["gsm8k"], cfg["test"]["mmlu"], cfg["seed"])
    assert_no_overlap(train, public)
    write_jsonl(public, pdir / "test_public.jsonl")
    backend = read_jsonl(pdir / "backend_set.jsonl")
    test = pd.concat([public, backend], ignore_index=True)
    print(test.groupby("dataset").size().to_string(), "\n(zero train/test overlap by ID and normalized text)\n")

    cache = ResponseCache()
    specs = specs_for(test, "haiku") + specs_for(test, "sonnet")
    est = estimate(cache.missing(specs), batch=True)
    print(format_estimate(est))
    check_budget(est)
    if args.estimate:
        print("\nNo calls made. Re-run with --run to submit.")
        return

    run_batch(specs, PHASE, cache)
    for mk in ("haiku", "sonnet"):
        df = graded(test, mk, cache)
        write_jsonl(df, path("responses_dir") / f"test_{mk}.jsonl")
        print(f"\n=== {mk}: {len(df)} graded ===")
        print(accuracy_report(df, ["dataset"]).to_string())
        print(accuracy_report(df[df.dataset == "backend"], ["category", "difficulty"]).to_string())
        print(f"stop=max_tokens: {(df.stop_reason == 'max_tokens').sum()}  refusals: {(df.stop_reason == 'refusal').sum()}")


if __name__ == "__main__":
    main()
