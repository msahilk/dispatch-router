"""Phase 0 smoke test: one tiny Haiku call (~$0.0001) to prove the API key works.

    uv run python scripts/phase0_smoke.py            # prints the estimate only
    uv run python scripts/phase0_smoke.py --run      # makes the call
"""

import argparse

from dispatch_router.llm import CallSpec, ResponseCache, estimate, format_estimate, run_sync

PROMPT = "What is 17 * 3? Reply with one short line ending in `ANSWER: <number>`."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually call the API")
    args = ap.parse_args()

    spec = CallSpec("haiku", PROMPT, max_tokens=64, dataset="smoke")
    cache = ResponseCache()
    pending = cache.missing([spec])
    print(format_estimate(estimate(pending, batch=False)))
    if not args.run:
        print("\nNo calls made. Re-run with --run to call the API.")
        return
    rec = run_sync([spec], phase="phase0_smoke", cache=cache)[0]
    print("\nResponse:", rec["text"].strip())
    print(f"model={rec['model_id']} in={rec['input_tokens']} out={rec['output_tokens']} "
          f"cost=${rec['cost_usd']:.6f} latency={rec['latency_s'] and round(rec['latency_s'], 2)}s "
          f"stop={rec['stop_reason']}")
    print("(cached: re-running costs $0)" if not pending else "")


if __name__ == "__main__":
    main()
