"""Anthropic client wrapper: record-once cache, cost tracking, budget guard, batches.

Every call is described by a `CallSpec`. Before paying for anything, callers:
  1. `pending = cache.missing(specs)`         -> drops calls we already paid for
  2. `est = estimate(pending)`; print it       -> pre-flight
  3. `check_budget(est)`                       -> abort if it would breach the cap
  4. `run_batch(pending, phase)` or `run_sync` -> results land in the cache + ledger
Then `cache.get(spec)` returns the stored record for every spec, at zero cost.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from .config import ROOT, load_config, path

CHARS_PER_TOKEN = 3.5  # conservative heuristic for pre-flight estimates only


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CallSpec:
    model_key: str  # "haiku" | "sonnet"
    prompt: str
    max_tokens: int
    dataset: str = ""  # used only to pick an expected-output size for estimates

    def params(self) -> dict[str, Any]:
        """The exact Messages API request body. Identical template for both models;
        only the per-model thinking-off setting differs (see config.yaml)."""
        m = load_config()["models"][self.model_key]
        return {
            "model": m["id"],
            "max_tokens": self.max_tokens,
            "thinking": dict(m["thinking"]),
            "output_config": {"effort": m["effort"]},
            "messages": [{"role": "user", "content": self.prompt}],
        }

    @property
    def key(self) -> str:
        """Cache key: sha256 of (model ID, prompt, params). 64 hex chars, which is
        also a valid Batches API custom_id."""
        blob = json.dumps(self.params(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- cost


def cost_usd(model_key: str, input_tokens: int, output_tokens: int, batch: bool) -> float:
    cfg = load_config()
    m = cfg["models"][model_key]
    usd = (input_tokens * m["price_input_per_mtok"] + output_tokens * m["price_output_per_mtok"]) / 1e6
    return usd * (cfg["batch_discount"] if batch else 1.0)


def estimate(specs: list[CallSpec], batch: bool = True) -> dict[str, Any]:
    cfg = load_config()
    margin = cfg["estimate_margin"]
    by_model: dict[str, dict[str, float]] = {}
    for s in specs:
        d = by_model.setdefault(s.model_key, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        d["calls"] += 1
        d["input_tokens"] += len(s.prompt) / CHARS_PER_TOKEN + 10
        d["output_tokens"] += cfg["expected_output_tokens"].get(s.dataset, s.max_tokens // 4)
    total = 0.0
    for mk, d in by_model.items():
        d["input_tokens"] = int(d["input_tokens"] * margin)
        d["output_tokens"] = int(d["output_tokens"] * margin)
        d["usd"] = cost_usd(mk, d["input_tokens"], d["output_tokens"], batch)
        # worst case: every call runs to max_tokens
        d["usd_worst_case"] = cost_usd(
            mk, d["input_tokens"], sum(s.max_tokens for s in specs if s.model_key == mk), batch
        )
        total += d["usd"]
    spent = Ledger().total()
    cap = cfg["budget"]["cap_usd"]
    return {
        "mode": "batch" if batch else "sync",
        "by_model": by_model,
        "calls": len(specs),
        "usd": total,
        "spent_so_far": spent,
        "cap": cap,
        "remaining_after": cap - spent - total,
    }


def format_estimate(est: dict[str, Any]) -> str:
    lines = [f"Pre-flight estimate ({est['mode']} mode, includes {load_config()['estimate_margin']}x margin)"]
    for mk, d in est["by_model"].items():
        lines.append(
            f"  {mk:<7} calls={d['calls']:>5}  in≈{d['input_tokens']:>9,} tok  out≈{d['output_tokens']:>9,} tok"
            f"  ≈ ${d['usd']:.4f}  (worst case if every answer hits max_tokens: ${d['usd_worst_case']:.4f})"
        )
    lines.append(f"  TOTAL calls={est['calls']}  ≈ ${est['usd']:.4f}")
    lines.append(
        f"  Spent so far: ${est['spent_so_far']:.4f} / cap ${est['cap']:.2f}"
        f"  ->  remaining after this run ≈ ${est['remaining_after']:.4f}"
    )
    return "\n".join(lines)


def check_budget(est: dict[str, Any]) -> None:
    if est["remaining_after"] < 0:
        raise BudgetExceeded(
            f"Run would cost ≈${est['usd']:.4f} but only ${est['cap'] - est['spent_so_far']:.4f} "
            f"of the ${est['cap']:.2f} cap remains. Aborting before any call."
        )


# ------------------------------------------------------------------------- ledger


class Ledger:
    FIELDS = ["timestamp", "phase", "model", "mode", "calls", "input_tokens", "output_tokens", "usd"]

    def __init__(self, file: Path | None = None):
        self.file = file or ROOT / load_config()["budget"]["ledger_path"]

    def total(self) -> float:
        if not self.file.exists():
            return 0.0
        with open(self.file) as f:
            return sum(float(r["usd"]) for r in csv.DictReader(f))

    def append(self, phase: str, model: str, mode: str, calls: int, inp: int, out: int, usd: float) -> None:
        if calls == 0:
            return
        new = not self.file.exists()
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "phase": phase,
                    "model": model,
                    "mode": mode,
                    "calls": calls,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "usd": f"{usd:.6f}",
                }
            )


# -------------------------------------------------------------------------- cache


class ResponseCache:
    """Append-only JSONL store keyed by CallSpec.key. Re-running never re-pays."""

    def __init__(self, file: Path | None = None):
        self.file = file or path("cache_dir") / "responses.jsonl"
        self._data: dict[str, dict[str, Any]] = {}
        if self.file.exists():
            with open(self.file) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec

    def __contains__(self, spec: CallSpec) -> bool:
        return spec.key in self._data

    def get(self, spec: CallSpec) -> dict[str, Any] | None:
        return self._data.get(spec.key)

    def missing(self, specs: Iterable[CallSpec]) -> list[CallSpec]:
        seen, out = set(), []
        for s in specs:
            if s.key not in self._data and s.key not in seen:
                seen.add(s.key)
                out.append(s)
        return out

    def put(self, rec: dict[str, Any]) -> None:
        self._data[rec["key"]] = rec
        with open(self.file, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------- client


def client():
    import anthropic

    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Put it in .env (see .env.example).")
    # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
    return anthropic.Anthropic(max_retries=load_config()["api"]["max_retries"])


def _record(spec: CallSpec, message: Any, mode: str, latency_s: float | None) -> dict[str, Any]:
    text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
    u = message.usage
    return {
        "key": spec.key,
        "model_key": spec.model_key,
        "model_id": message.model,
        "mode": mode,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost_usd": cost_usd(spec.model_key, u.input_tokens, u.output_tokens, batch=(mode == "batch")),
        "latency_s": latency_s,  # None for batch: batches give no real per-request latency
        "stop_reason": message.stop_reason,
        "text": text,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _log_spend(phase: str, recs: list[dict[str, Any]], mode: str) -> None:
    ledger = Ledger()
    for mk in sorted({r["model_key"] for r in recs}):
        rs = [r for r in recs if r["model_key"] == mk]
        ledger.append(
            phase,
            rs[0]["model_id"],
            mode,
            len(rs),
            sum(r["input_tokens"] for r in rs),
            sum(r["output_tokens"] for r in rs),
            sum(r["cost_usd"] for r in rs),
        )


def run_sync(specs: list[CallSpec], phase: str, cache: ResponseCache | None = None) -> list[dict[str, Any]]:
    """Normal Messages API calls, one at a time. For small runs (smoke test)."""
    cache = cache or ResponseCache()
    pending = cache.missing(specs)
    check_budget(estimate(pending, batch=False))
    c = client()
    recs = []
    try:
        for s in pending:
            t0 = time.perf_counter()
            msg = c.messages.create(**s.params())
            rec = _record(s, msg, "sync", time.perf_counter() - t0)
            cache.put(rec)
            recs.append(rec)
    finally:
        _log_spend(phase, recs, "sync")
    return [cache.get(s) for s in specs]


def run_batch(specs: list[CallSpec], phase: str, cache: ResponseCache | None = None) -> list[dict[str, Any]]:
    """Message Batches API (50% off). Batch IDs are saved to data/batches/ so an
    interrupted run resumes by polling the existing batch instead of resubmitting."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    cfg = load_config()
    cache = cache or ResponseCache()
    pending = cache.missing(specs)
    by_key = {s.key: s for s in pending}
    state_file = path("batches_dir") / f"{phase}.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"batches": []}

    c = client()
    # Resume: any still-recorded batch whose requests we have not cached yet.
    open_batches = [b for b in state["batches"] if not b.get("collected")]
    if not open_batches and pending:
        check_budget(estimate(pending, batch=True))
        batch = c.messages.batches.create(
            requests=[Request(custom_id=s.key, params=MessageCreateParamsNonStreaming(**s.params())) for s in pending]
        )
        state["batches"].append({"id": batch.id, "n": len(pending), "created": batch.created_at.isoformat()})
        state_file.write_text(json.dumps(state, indent=2))
        print(f"Submitted batch {batch.id} with {len(pending)} requests (saved to {state_file}).")
        open_batches = [state["batches"][-1]]

    for b in open_batches:
        while True:
            info = c.messages.batches.retrieve(b["id"])
            rc = info.request_counts
            print(
                f"  batch {b['id']}: {info.processing_status} "
                f"(processing={rc.processing} succeeded={rc.succeeded} errored={rc.errored})",
                flush=True,
            )
            if info.processing_status == "ended":
                break
            time.sleep(cfg["api"]["batch_poll_seconds"])
        recs, failed = [], []
        for r in c.messages.batches.results(b["id"]):
            spec = by_key.get(r.custom_id)
            if spec is None:  # already cached, or belongs to a spec no longer requested
                continue
            if r.result.type == "succeeded":
                rec = _record(spec, r.result.message, "batch", None)
                cache.put(rec)
                recs.append(rec)
            else:
                failed.append((r.custom_id, r.result.type))
        _log_spend(phase, recs, "batch")
        b["collected"] = True
        state_file.write_text(json.dumps(state, indent=2))
        print(f"  collected {len(recs)} results; {len(failed)} not succeeded")
        if failed:
            print("  Re-run the same command to resubmit the failed requests.")
    return [cache.get(s) for s in specs]
