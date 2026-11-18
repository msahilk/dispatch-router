"""Build the public training pool and label it with Haiku (right / wrong).

    uv run python scripts/phase1_train_data.py --estimate   # builds the pool, prints cost, no calls
    uv run python scripts/phase1_train_data.py --run        # submits a Message Batch to Haiku
"""

import argparse

from dispatch_router.config import ROOT, path
from dispatch_router.datasets import build_train_pool, write_jsonl
from dispatch_router.llm import ResponseCache, check_budget, estimate, format_estimate, run_batch
from dispatch_router.runs import accuracy_report, graded, specs_for

PHASE = "train_haiku"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()

    pool = build_train_pool()
    write_jsonl(pool, path("prompts_dir") / "train_pool.jsonl")
    print(pool.groupby(["dataset", "split"]).size().unstack().to_string(), "\n")

    cache = ResponseCache()
    specs = specs_for(pool, "haiku")
    pending = cache.missing(specs)
    est = estimate(pending, batch=True)
    print(format_estimate(est))
    check_budget(est)
    if args.estimate:
        print("\nNo calls made. Re-run with --run to submit.")
        return

    if pending:
        run_batch(specs, PHASE, cache)
    df = graded(pool, "haiku", cache)
    write_jsonl(df, path("responses_dir") / "haiku_train.jsonl")
    print(f"\nLabeled {len(df)}/{len(pool)} items -> data/responses/haiku_train.jsonl\n")
    print(accuracy_report(df, ["dataset"]).to_string())
    print(f"\nOverall Haiku accuracy {df.correct.mean():.3f}; parse-fail rate {df.parse_fail.mean():.3%}")
    print(f"Hit max_tokens: {(df.stop_reason == 'max_tokens').sum()}  refusals: {(df.stop_reason == 'refusal').sum()}")


if __name__ == "__main__":
    main()
