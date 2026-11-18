"""Build results/summary.json, the README and draft resume notes from result files.

Every number in README.md and RESUME_NOTES.md is read from files this project
produced (results/tables/*, results/honesty.json, results/spend_ledger.csv).
Nothing is typed by hand.

    uv run python scripts/phase9_report.py
"""

import json

import pandas as pd

from dispatch_router import plots
from dispatch_router.config import ROOT, load_config, path
from dispatch_router.llm import Ledger
from dispatch_router.scoring import train_pool

ROUTERS = ["logreg", "distilbert", "knn", "prior"]
NAMES = {"logreg": "Logistic regression", "distilbert": "DistilBERT", "knn": "kNN",
         "prior": "Subject prior (no NLP)", "random": "Random", "oracle": "Oracle"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def ci(lo, hi, d=1, scale=100, suffix="%"):
    return f"[{scale * lo:.{d}f}, {scale * hi:.{d}f}]{suffix}"


def main() -> None:
    cfg = load_config()
    res_dir, tables = path("results_dir"), path("results_dir") / "tables"
    res = pd.read_csv(tables / "results_all.csv")
    cal = pd.read_csv(tables / "calibration.csv")
    thr = pd.read_csv(tables / "threshold_control.csv")
    hon = json.loads((res_dir / "honesty.json").read_text())
    hist = pd.read_csv(tables / "distilbert_history.csv")
    spend = pd.read_csv(Ledger().file)
    plots.train_val_curves(hist.to_dict("records"), res_dir / "figures" / "distilbert_curves.png")

    def row(s, r):
        return res[(res.set == s) & (res.router == r)].iloc[0]

    lr, rnd, orc = row("public", "logreg"), row("public", "random"), row("public", "oracle")
    lrb, rndb = row("backend", "logreg"), row("backend", "random")
    db = row("public", "distilbert")
    vs = pd.DataFrame(hon["vs_logreg"])
    d_vs = vs[(vs.comparison == "distilbert - logreg")].set_index(["set", "metric"])
    of = pd.DataFrame(hon["overfitting"]).set_index("router")
    gen = pd.DataFrame(hon["generalization"]).set_index("router")
    sanity = pd.DataFrame(hon["sanity"])
    cats = pd.DataFrame(hon["backend_categories"])
    n = {s: int(row(s, "random").n) for s in ("public", "gsm8k", "mmlu", "backend", "backend_stress")}
    pool = train_pool()
    train_n = len(pool)
    tr_rows = pool[pool.split == "train"]
    n_tr, n_tr_neg = len(tr_rows), int((~tr_rows.correct.astype(bool)).sum())
    n_by_ds = pool.dataset.value_counts().to_dict()

    summary = {
        "models": {k: v["id"] for k, v in cfg["models"].items()},
        "price_ratio_sonnet_over_haiku_output": cfg["models"]["sonnet"]["price_output_per_mtok"]
        / cfg["models"]["haiku"]["price_output_per_mtok"],
        "n_train_pool": train_n,
        "n_test": n,
        "main_router": "logreg",
        "public_test": {
            "acc_haiku": lr.acc_haiku, "acc_sonnet": lr.acc_sonnet,
            "haiku_cost_pct_of_sonnet": lr.haiku_cost_pct,
            "apgr": [lr.apgr, lr.apgr_lo, lr.apgr_hi],
            "cpt50": [lr.cpt50, lr.cpt50_lo, lr.cpt50_hi],
            "cpt80": [lr.cpt80, lr.cpt80_lo, lr.cpt80_hi],
            "cost_pct_at_99pct_of_sonnet_acc": [lr.cost_pct_at_99, lr.cost_pct_at_99_lo, lr.cost_pct_at_99_hi],
            "random_cost_pct_at_99pct_of_sonnet_acc": rnd.cost_pct_at_99,
            "cost_pct_at_95pct_of_sonnet_acc": [lr.cost_pct_at_95, lr.cost_pct_at_95_lo, lr.cost_pct_at_95_hi],
            "acc_at_30pct_sonnet": [lr.acc_at_30, lr.acc_at_30_lo, lr.acc_at_30_hi],
            "acc_kept_vs_sonnet_at_30pct": lr.acc_at_30 / lr.acc_sonnet,
            "cost_pct_at_30pct_sonnet": [lr.cost_pct_at_30, lr.cost_pct_at_30_lo, lr.cost_pct_at_30_hi],
            "auroc": [lr.auroc, lr.auroc_lo, lr.auroc_hi],
            "random_apgr": rnd.apgr, "oracle_apgr": orc.apgr,
        },
        "backend_set": {
            "acc_haiku": lrb.acc_haiku, "acc_sonnet": lrb.acc_sonnet,
            "apgr": [lrb.apgr, lrb.apgr_lo, lrb.apgr_hi], "auroc": [lrb.auroc, lrb.auroc_lo, lrb.auroc_hi],
            "cost_pct_at_99pct_of_sonnet_acc": lrb.cost_pct_at_99,
            "random_cost_pct_at_99pct_of_sonnet_acc": rndb.cost_pct_at_99,
        },
        "distilbert_minus_logreg": {
            "public_apgr": [d_vs.loc[("public", "apgr")]["diff"], d_vs.loc[("public", "apgr")].ci_lo,
                            d_vs.loc[("public", "apgr")].ci_hi],
            "backend_apgr": [d_vs.loc[("backend", "apgr")]["diff"], d_vs.loc[("backend", "apgr")].ci_lo,
                             d_vs.loc[("backend", "apgr")].ci_hi],
        },
        "overfitting_auroc": of[["auroc_train", "auroc_val", "auroc_test_public", "auroc_backend"]].to_dict("index"),
        "total_api_spend_usd": float(spend.usd.sum()),
        "bootstrap_resamples": 1000,
    }
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---------------------------------------------------------------- results table
    ece = cal.set_index(["router", "set"]).ece_cal
    lines = ["| Test set | Router | APGR [95% CI] | CPT(50%) | CPT(80%) | Cost @ 95% of Sonnet | "
             "Cost @ 99% of Sonnet | AUROC | ECE (calibrated) |", "|---|---|---|---|---|---|---|---|---|"]
    for s, label in (("public", "Public"), ("mmlu", "MMLU"), ("gsm8k", "GSM8K"), ("backend", "Backend")):
        for r in ["random", "oracle"] + ROUTERS:
            x = row(s, r)
            a = f"{x.apgr:.3f}" + (f" [{x.apgr_lo:.2f}, {x.apgr_hi:.2f}]" if r in ROUTERS else "")
            au = f"{x.auroc:.3f}" if r in ROUTERS else "—"
            e_set = {"public": "test_public", "backend": "backend"}.get(s)
            e = f"{ece[(r, e_set)]:.3f}" if r in ROUTERS and e_set else "—"
            nm = f"**{NAMES[r]}**" if r == "logreg" else NAMES[r]
            lines.append(f"| {label} | {nm} | {a} | {pct(x.cpt50)} | {pct(x.cpt80)} | {x.cost_pct_at_95:.1f}% | "
                         f"{x.cost_pct_at_99:.1f}% | {au} | {e} |")
    results_table = "\n".join(lines)

    acc_lines = ["| Test set | n | Haiku | Sonnet | Only Sonnet right | Only Haiku right | Haiku cost (% of Sonnet) |",
                 "|---|---|---|---|---|---|---|"]
    for s, label in (("gsm8k", "GSM8K"), ("mmlu", "MMLU"), ("backend", "Backend (all)"),
                     ("backend_stress", "Backend (stress tier)")):
        x = row(s, "random")
        acc_lines.append(f"| {label} | {int(x.n)} | {pct(x.acc_haiku)} | {pct(x.acc_sonnet)} | "
                         f"{int(x.n_only_sonnet_right)} | {int(x.n_only_haiku_right)} | {x.haiku_cost_pct:.1f}% |")
    acc_table = "\n".join(acc_lines)

    fin = cats[(cats.category == "finance") & (cats.tier == "stress")].iloc[0]
    pys = cats[(cats.category == "python") & (cats.tier == "stress")].iloc[0]
    t20 = thr[(thr.router == "logreg") & (thr.target_sonnet_rate == 0.2)].iloc[0]
    dist_ece_raw = cal[(cal.router == "distilbert") & (cal.set == "test_public")].iloc[0]
    sp = summary["public_test"]

    readme = f"""# dispatch-router

**Decide, before calling an LLM, whether a prompt can go to the cheap model (Claude Haiku 5.5) or needs the
expensive one (Claude Sonnet 5.5) — then measure honestly how much that saves and what it costs in quality.**

On {n['public']:,} held-out GSM8K + MMLU questions, a logistic-regression router over prompt embeddings reaches
**99% of Sonnet's accuracy at {sp['cost_pct_at_99pct_of_sonnet_acc'][0]:.1f}% of always-Sonnet cost**
(95% CI {ci(*sp['cost_pct_at_99pct_of_sonnet_acc'][1:], scale=1)}; random routing needs
{sp['random_cost_pct_at_99pct_of_sonnet_acc']:.1f}%). It recovers half of the Haiku→Sonnet quality gap while
sending **{pct(sp['cpt50'][0])}** of calls to Sonnet (CI {ci(*sp['cpt50'][1:])}) and 80% while sending
**{pct(sp['cpt80'][0])}** (CI {ci(*sp['cpt80'][1:])}); its APGR is **{sp['apgr'][0]:.3f}**
(CI [{sp['apgr'][1]:.3f}, {sp['apgr'][2]:.3f}]) against 0.5 for random. A fine-tuned DistilBERT does no better
and generalizes worse. Total API spend: **${summary['total_api_spend_usd']:.2f}**.

![Cost-quality curves](results/figures/cost_quality.png)

*Each line sweeps the routing threshold from "everything to Haiku" (left) to "everything to Sonnet" (right).
Above the dashed diagonal = better than random routing; the dotted line is the oracle (perfect foresight).*

## Setup

| | |
|---|---|
| Models | `{summary['models']['haiku']}` (cheap) vs `{summary['models']['sonnet']}` (≈{summary['price_ratio_sonnet_over_haiku_output']:.0f}× the price); thinking off, identical prompts, Message Batches API |
| Training labels | {train_n:,} public questions ({n_by_ds["gsm8k"]:,} GSM8K train + all {n_by_ds["mmlu"]:,} MMLU val/dev), each labeled "did Haiku get it right?" — 85/15 train/validation split |
| Test sets | {n['public']:,} public ({n["gsm8k"]:,} GSM8K test + {n["mmlu"]:,} MMLU test, zero overlap with training) and a {n['backend']}-question **backend-engineering set** (Python output, SQL, infra/reliability math, finance math) with gold answers computed by executing the code/SQL/formula |
| Routers | random (baseline), oracle (upper bound), subject prior (Haiku's training accuracy for the topic, no language model), kNN and logistic regression on `bge-small-en-v1.5` embeddings, fine-tuned DistilBERT (Apple MPS) |
| Calibration | Platt scaling fit on validation only |
| Uncertainty | {summary['bootstrap_resamples']:,} bootstrap resamples of the test prompts; 95% percentile CIs |

Every router outputs p(Haiku answers correctly); a prompt goes to Haiku if the score clears a threshold τ.

**Metrics.** *PGR* (performance gap recovered) = (router accuracy − Haiku accuracy) / (Sonnet accuracy − Haiku
accuracy). *APGR* = area under PGR vs. share of calls sent to Sonnet (random = 0.5). *CPT(x%)* = smallest share of
Sonnet calls that recovers x% of the gap. *Cost @ y% of Sonnet* = cheapest point whose accuracy is ≥ y% of
Sonnet's, as a % of always-Sonnet cost. *AUROC* = how well the score ranks Haiku's failures below its successes
(0.5 = chance). *ECE* = average gap between predicted and actual Haiku accuracy across 10 score bins.

## Model accuracy on the test sets

{acc_table}

## Results

{results_table}

![PGR curves](results/figures/pgr.png)

## What worked / what didn't

- **Routing beats random, modestly.** Logistic regression's APGR is {sp['apgr'][0]:.3f} vs 0.5 on the public test.
  Predicting which prompts a strong cheap model will miss is hard: the oracle sits far above every router
  (APGR {sp['oracle_apgr']:.2f}).
- **Most of the signal is topic difficulty.** A lookup of Haiku's training accuracy per MMLU subject — no language
  model — gets APGR {row('public', 'prior').apgr:.3f}. Embeddings add a real but small edge over it.
- **Simple beat complex.** DistilBERT minus logistic regression: APGR
  {summary['distilbert_minus_logreg']['public_apgr'][0]:+.3f} on the public test
  (CI [{summary['distilbert_minus_logreg']['public_apgr'][1]:+.3f}, {summary['distilbert_minus_logreg']['public_apgr'][2]:+.3f}], no
  difference) and {summary['distilbert_minus_logreg']['backend_apgr'][0]:+.3f} on the backend set
  (CI [{summary['distilbert_minus_logreg']['backend_apgr'][1]:+.3f}, {summary['distilbert_minus_logreg']['backend_apgr'][2]:+.3f}]).
  DistilBERT overfits {n_tr:,} training examples containing only {n_tr_neg} "Haiku wrong" cases: train AUROC
  {of.loc['distilbert', 'auroc_train']:.2f} vs validation {of.loc['distilbert', 'auroc_val']:.2f}
  (logistic regression: {of.loc['logreg', 'auroc_train']:.2f} vs {of.loc['logreg', 'auroc_val']:.2f}).
- **Generalization to backend prompts.** Logistic regression keeps most of its value (APGR
  {gen.loc['logreg', 'apgr_public']:.3f} → {gen.loc['logreg', 'apgr_backend']:.3f}). kNN
  ({gen.loc['knn', 'apgr_backend']:.3f}, AUROC {gen.loc['knn', 'auroc_backend']:.2f}) and DistilBERT
  ({gen.loc['distilbert', 'apgr_backend']:.3f}, AUROC {gen.loc['distilbert', 'auroc_backend']:.2f}) fall *below
  random*: they are most confident on multi-step finance questions that resemble GSM8K word problems — where Haiku
  is {pct(fin.acc_haiku)} accurate vs Sonnet {pct(fin.acc_sonnet)}.
- **"Looks easy but isn't."** At a 30% Sonnet budget the router sends {pct(pys.sonnet_share_at_30, 0)} of
  stress-tier Python snippets to Sonnet, though Haiku misses {int(pys.n_only_sonnet_right)} that Sonnet gets right.
- **Classic gotchas no longer separate the models.** Mutable defaults, late-binding closures, `NOT IN` with NULLs
  and similar traps were answered correctly by both models; only the harder stress tier opened a gap.
- **Calibration.** Scores from logistic regression were already close to calibrated (public-test ECE
  {cal[(cal.router == 'logreg') & (cal.set == 'test_public')].ece_raw.iloc[0]:.3f} raw →
  {cal[(cal.router == 'logreg') & (cal.set == 'test_public')].ece_cal.iloc[0]:.3f} after Platt). DistilBERT's class-weighted
  training made it badly miscalibrated (ECE {dist_ece_raw.ece_raw:.3f}), which Platt scaling fixed
  ({dist_ece_raw.ece_cal:.3f}). ![Reliability](results/figures/reliability_public_test.png)
- **The threshold knob drifts under shift.** A τ that sends 20% of validation prompts to Sonnet sends
  {pct(t20.realized_test_public, 0)} of public-test and {pct(t20.realized_backend, 0)} of backend prompts there,
  because those sets are harder than validation. Re-tune τ on live traffic.
- **Sanity checks pass.** The measured random router averages APGR
  {sanity.random_apgr_measured_mean_200.min():.3f}–{sanity.random_apgr_measured_mean_200.max():.3f} (expected 0.5) and no router
  beats the oracle on any test set.
- **The PRD's headline target was too easy.** "95% of Sonnet's accuracy" is met by always using Haiku
  ({sp['haiku_cost_pct_of_sonnet']:.1f}% of the cost), so this README leads with 99%.

More: [backend categories](results/figures/backend_categories.png), [DistilBERT train vs validation](results/figures/distilbert_curves.png),
[more curves](results/figures/cost_quality_more.png), all tables in [`results/tables/`](results/tables/).

## Limitations

- One model pair and one snapshot of each; results may not transfer to other pairs.
- Thinking was turned off and temperature can't be set on these models, so answers aren't bit-for-bit
  repeatable; each answer was recorded once and every number replays those recordings.
- Only {int(row('public', 'random').n_only_sonnet_right)} public and {int(row('backend', 'random').n_only_sonnet_right)} backend prompts are "only Sonnet right", so CIs (especially on the backend set) are wide.
- GSM8K and MMLU may appear in the models' training data; ~1–2% of GSM8K gold answers are known to be wrong.
- The backend set is synthetic and template-based; several items share a template with different numbers.
- Latency is not reported: batch calls don't expose per-request latency.

## Reproduce

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY (only needed to re-record answers)
uv run python scripts/phase0_smoke.py --run          # ~$0.0003
uv run python scripts/phase1_train_data.py --run     # Haiku labels (cached in data/responses)
uv run python scripts/phase2_build_backend_set.py    # free; verifies answers by execution
uv run python scripts/phase2_test_data.py --run      # both models on test sets (cached)
uv run python scripts/phase3_embed.py                # free, local
uv run python scripts/phase4_train_routers.py        # includes DistilBERT on MPS (~20 min)
uv run python scripts/phase5_calibrate.py
uv run python scripts/phase6_evaluate.py
uv run python scripts/phase7_honesty.py
uv run python scripts/phase9_report.py               # summary.json, this README
uv run pytest
```

Recorded responses are committed, so every step after Phase 2 runs offline and costs $0. Re-running a paid
script with the cache in place also costs $0.

## API spend

${summary['total_api_spend_usd']:.2f} across {len(spend)} runs ({int(spend.calls.sum()):,} calls), itemized in
[`results/spend_ledger.csv`](results/spend_ledger.csv). Approach follows RouteLLM (Ong et al., ICLR 2025).
"""
    (ROOT / "README.md").write_text(readme)

    resume = f"""# Resume notes (DRAFT — edit before using)

All numbers come from `results/summary.json`.

**Long bullet**
Built a calibrated LLM router that decides before each call whether Claude Haiku can handle a prompt or it needs
Claude Sonnet; on {n['public']:,} held-out benchmark questions it matched 99% of Sonnet's accuracy at
{sp['cost_pct_at_99pct_of_sonnet_acc'][0]:.0f}% of the cost (random routing: {sp['random_cost_pct_at_99pct_of_sonnet_acc']:.0f}%),
evaluated with bootstrap CIs, a {n['backend']}-question execution-verified backend-engineering test set, and
comparisons showing a logistic-regression router generalizes better than fine-tuned DistilBERT.

**Short bullet**
LLM cost router (Haiku vs Sonnet): 99% of Sonnet accuracy at {sp['cost_pct_at_99pct_of_sonnet_acc'][0]:.0f}% of cost on
{n['public']:,} held-out prompts; ${summary['total_api_spend_usd']:.2f} total API spend.

**~300-character paragraph**
I built a router that sends each prompt to a cheap or an expensive Claude model before calling either. Trained on
{train_n:,} labeled questions, it kept 99% of the expensive model's accuracy at {sp['cost_pct_at_99pct_of_sonnet_acc'][0]:.0f}%
of the cost, and I tested where it breaks on my own backend-engineering prompt set.
"""
    (res_dir / "RESUME_NOTES.md").write_text(resume)
    print(json.dumps(summary["public_test"], indent=1, default=float))
    print(f"\nREADME.md, results/summary.json, results/RESUME_NOTES.md written. Spend ${summary['total_api_spend_usd']:.4f}")


if __name__ == "__main__":
    main()
