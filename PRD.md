# PRD: Calibrated LLM Router (Claude Haiku vs Sonnet)

_Working name: **llm-router**. Rename freely._

## 0. How to use this document (instructions for Claude Code)

You are building this project with the user, Sahil. He is a backend engineer and new to ML, so:

- **Work in the phases below, in order.** Finish a phase, show what you built and the key numbers, then move on.
- **Stop and ask Sahil** at every point marked **🛑 CHECKPOINT**. Tell him exactly what to do (commands to run, what to check), wait for him to confirm, then continue.
- **Never spend money without asking.** Before any paid API run, print a cost estimate and wait for an explicit "yes".
- **Explain as you go.** After each phase, give a short plain-English summary: what this step does, why it matters, and what the result means. Avoid unexplained jargon.
- **Be honest in results.** If something doesn't work (a router doesn't beat the baseline, DistilBERT loses to logistic regression), report it plainly. Negative results are part of the deliverable.
- **Don't invent numbers.** Every number in the README must come from a file this project produced.
- Look up anything that may have changed since your training (model IDs, prices, SDK details) in the official docs before relying on it.

## 1. Goal

Build a **predictive router** that decides, **before calling any model**, whether a prompt should go to a cheap model (**Claude Haiku**) or an expensive one (**Claude Sonnet**). Then measure how much cost it saves and how much quality it loses, compared against honest baselines.

The approach follows **RouteLLM** (LMSYS, ICLR 2025): train routers on public data, calibrate them, and evaluate on cost-quality curves with standard metrics. This project adds:
1. labels from the actual cheap model being used (Haiku),
2. a fine-tuned DistilBERT router compared against simpler routers,
3. a custom **backend-engineering prompt set** to test generalization.

### Success looks like

- A cost-quality curve where at least one router clearly beats the random baseline, **or** a clear, well-documented finding that it doesn't.
- Headline numbers, each with a 95% confidence interval:
  - accuracy kept versus all-Sonnet;
  - cost versus all-Sonnet;
  - CPT(50%), CPT(80%) and APGR.
- A reproducible repo with a README a recruiter can skim in 2 minutes.

### Out of scope (do not build)

Semantic caching, cascades (escalate-after-answer), a Go serving layer, multi-provider failover, and any production deployment.

## 2. Constraints

| Item | Constraint |
|---|---|
| Hardware | MacBook Air M4 (Apple Silicon). Use PyTorch's **MPS** backend for DistilBERT; CPU for everything else |
| API budget | **Hard cap: US$5 total** across all paid runs. Track spend in a ledger file |
| Provider | Anthropic only (one key) |
| Language | Python 3.11+ |
| Time | Prefer simple, standard implementations over clever ones |

## 3. Tech stack

- Package and env management: `uv` (or `venv` + `pip` if `uv` is unavailable). Pin versions in `pyproject.toml` / lockfile.
- Libraries:
  - `anthropic` (official SDK);
  - `datasets` (Hugging Face, for GSM8K and MMLU);
  - `sentence-transformers` (embeddings);
  - `scikit-learn` (kNN, logistic regression, calibration, metrics);
  - `torch` and `transformers` (DistilBERT);
  - `numpy`, `pandas`, `matplotlib`;
  - `pytest`;
  - Python's built-in `sqlite3` (for verifying SQL prompts).
- Config: one `config.yaml` holding model IDs, prices per million tokens, sample sizes, seeds, max tokens and budget cap.
- Secrets: `ANTHROPIC_API_KEY` from the environment or a gitignored `.env`. Never commit or print it.

## 4. Repository layout

```
llm-router/
  README.md
  PRD.md                      # this file
  config.yaml
  pyproject.toml
  .env.example                # ANTHROPIC_API_KEY=
  data/
    raw/                      # downloaded benchmark data (gitignored if large)
    prompts/                  # built prompt sets (jsonl)
    responses/                # recorded model outputs (jsonl), the "record once" store
    embeddings/               # cached .npy
  src/llm_router/
    datasets.py               # load + sample GSM8K / MMLU, build splits
    backend_set.py            # build + verify the backend prompt set
    prompts.py                # prompt templates + answer extraction + grading
    llm.py                    # Anthropic client: caching, retries, batches, cost tracking
    embed.py                  # embeddings
    routers/                  # random, always, oracle, knn, logreg, distilbert
    calibrate.py
    evaluate.py               # threshold sweep, metrics, bootstrap
    plots.py
  scripts/                    # one CLI entry point per phase (see §6)
  results/
    figures/                  # png charts
    tables/                   # csv/markdown tables
    summary.json              # headline numbers for README/resume
    spend_ledger.csv          # every paid call batch: date, phase, model, tokens, USD
  tests/
```

## 5. Shared rules for all API calls (`llm.py`)

1. **Record once, replay many.** Cache every response on disk, keyed by a hash of (model ID, prompt, params). Re-running a script must never pay for the same call twice.
2. **Cost tracking.** Compute cost from the API's reported input and output token counts and the prices in `config.yaml`. Append each run's totals to `results/spend_ledger.csv`.
3. **Pre-flight estimate.** Before any paid run, print: number of calls, estimated tokens, estimated USD, and remaining budget. 🛑 Wait for Sahil's "yes".
4. **Budget guard.** Abort before starting a run that would push total spend over the cap.
5. **Prefer the Message Batches API** for bulk runs. It's roughly 50% cheaper and results can take a while, which is fine here. Fall back to normal calls with retries and exponential backoff for small runs. Save batch IDs so an interrupted run can resume.
6. **Deterministic settings:** temperature 0, the same prompt template for both models, and a fixed `max_tokens` per dataset (in config).
7. **Record per call:** model ID, input tokens, output tokens, cost, latency (for non-batch calls; batches don't give real per-request latency, so note that in results), the raw output, the extracted answer and correctness.

## 6. Phases

### Phase 0: Setup

- Create the repo structure, `pyproject.toml`, `config.yaml`, `.env.example`, `.gitignore` and a basic test.
- Look up the **current Claude Haiku and Sonnet model IDs and prices** in Anthropic's docs. Put them in `config.yaml`, and show them to Sahil.
- 🛑 **CHECKPOINT:** ask Sahil to:
  1. Create `.env` with his `ANTHROPIC_API_KEY`.
  2. Confirm the model IDs and prices.
  3. Run a one-call smoke test (about $0.001) to prove the key works.

### Phase 1: Training data (public, Haiku-labeled, target ≤ ~$1)

**Plain English:** collect questions with known answers, ask Haiku each one, and record whether it got it right. The router learns from these "Haiku right / Haiku wrong" labels. Sonnet isn't needed here.

- **Sources** (via Hugging Face `datasets`):
  - **GSM8K** `train` split: sample ~1,000 (seeded).
  - **MMLU** `validation` + `dev` splits: ~1,500 across subjects (seeded, stratified by subject where practical).
- **Prompt templates** (in `prompts.py`):
  - GSM8K: ask for brief reasoning, ending with a final line `ANSWER: <number>`. Grade by normalized numeric equality (strip commas, `$`, units; compare as numbers).
  - MMLU: show the question and options A–D, allow brief reasoning, end with `ANSWER: <letter>`. Grade by exact letter.
  - If the answer line is missing, mark the result as incorrect and log it as `parse_fail`. Report the parse-fail rate.
- **Splits:** split the labeled pool into **train (85%) / validation (15%)**, seeded and stratified by dataset. Validation is used for tuning and calibration only.
- **Outputs:**
  - `data/prompts/train_pool.jsonl` (id, dataset, subject/category, prompt, gold answer, split);
  - `data/responses/haiku_train.jsonl`.
- 🛑 **CHECKPOINT:** show the pre-flight cost estimate and get approval. After the run, report Haiku's accuracy per dataset and the parse-fail rate.

### Phase 2: Test sets (both models, target ≤ ~$4)

**Plain English:** build questions the routers never see in training, and get **both** models' answers once. Every router is graded against these saved answers.

**2a. Public test set:** ~150 from GSM8K `test` and ~150 from MMLU `test`, seeded. Assert **zero overlap** with the training pool (by ID and by normalized text).

**2b. Backend-engineering prompt set (Sahil's custom set):** ~150–200 prompts covering a backend engineer's day. **Every item must have an exact, machine-checkable answer.** Categories:

| Category | Easy examples | Hard examples | Ground truth computed by |
|---|---|---|---|
| Python behavior ("what does this print?") | `sorted`, `split`, slicing | Mutable default args, late-binding closures, `-7 // 2`, generator exhaustion, `try/finally` return, dict key collisions (`1`, `True`, `1.0`) | Executing the snippet in a subprocess with a timeout |
| SQL on a given small table | Counts, filters | GROUP BY + HAVING, NULL traps (`!=` excludes NULL), "all orders paid", second-highest | Running the query in `sqlite3` |
| Infra and reliability math | Unit conversions, requests per minute | Pods per node after reserved CPU, SLO error budget minutes, retry backoff totals, fan-out tail probability, Little's law, Kafka partitions per consumer | Python arithmetic |
| Finance and ledger math | Percentages, simple interest | Compound interest, margin and position math, rounding to cents | Python arithmetic (Decimal) |

- **Target mix:** ~60% easy, ~40% hard, roughly balanced across categories. Tag each item `difficulty` and `category`.
- **How to build it:**
  1. Draft the items yourself (Claude Code).
  2. **Compute every gold answer by actually running the code, SQL or math.** Never trust a written answer.
  3. Drop any item where execution fails or the answer is ambiguous.
- **Answer format:** same `ANSWER: <x>` convention. Define normalization per category; for example, for printed output compare the exact string after stripping trailing whitespace.
- 🛑 **CHECKPOINT:** show Sahil a summary table (counts by category and difficulty) plus ~15 random samples with their verified answers, and ask him to skim and approve before any API calls.

**2c. Record both models.** Run all test prompts (2a + 2b) through **Haiku and Sonnet**, with identical prompts and settings. Save to `data/responses/test_haiku.jsonl` and `test_sonnet.jsonl`.

- 🛑 **CHECKPOINT:** pre-flight cost estimate and approval. Afterwards, report per-model accuracy by dataset, category and difficulty, the parse-fail rate and total spend.

### Phase 3: Embeddings (free, local)

**Plain English:** turn every prompt into a list of numbers where similar meanings get similar numbers. Routers 2 and 3 work from these.

- Use a small, standard `sentence-transformers` model (for example `all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`; pick one and record it in config).
- Embed all train, validation and test prompts, normalize them, and cache them to `data/embeddings/`.
- Briefly explain to Sahil what an embedding is (2–3 sentences).

### Phase 4: Routers

**Plain English:** each router reads a prompt and outputs a score: the probability that Haiku answers it correctly. High score means Haiku; low score means Sonnet.

Common interface: `fit(train, val)` and `score(prompts) -> p_haiku_correct ∈ [0,1]`. Routing rule: send to Haiku if the score ≥ threshold τ, else Sonnet.

| Router | Definition | Notes |
|---|---|---|
| `always_haiku`, `always_sonnet` | Fixed choice | The floor and the ceiling |
| `random` | Sends a fraction *f* of traffic to Sonnet at random, swept over *f* ∈ [0, 1] | **The baseline every router must beat.** Compute its expected curve analytically or average many seeds |
| `oracle` | Sends to Haiku exactly when the recorded Haiku answer was correct | Upper bound; uses test labels, so it's for reference only |
| `knn` | Score = (distance-weighted) share of the *k* nearest training prompts Haiku got right | Tune *k* on validation (e.g. 5–50) |
| `logreg` | Logistic regression on embeddings | **Main router.** Tune regularization C on validation; use class weights if labels are imbalanced |
| `distilbert` | Fine-tune `distilbert-base-uncased` for binary classification (Haiku correct?) | See below |

**DistilBERT settings** (starting point; adjust if justified and explain why):
- Device `mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Max length 256; batch size 16; learning rate 2e-5; 3–4 epochs; weight decay 0.01; fixed seed.
- **Early stopping on validation** (keep the best checkpoint by validation log-loss or AUROC).
- Log train versus validation curves per epoch so overfitting is visible.
- If MPS is unavailable or broken, fall back to CPU and say so.

### Phase 5: Calibration

**Plain English:** check whether "80% sure" really means about 80% right, and correct it if not.

- Fit calibration on the **validation** split only (Platt / sigmoid, or isotonic if validation is large enough; justify the choice). Apply it to the test set.
- Report **before versus after** for each learned router: Brier score, Expected Calibration Error (ECE, 10 bins) and a **reliability diagram** (predicted versus actual).
- Show **threshold calibration**: for target Sonnet-call rates of 10/20/30/50%, choose τ on validation and report the realized Sonnet-call rate on test. This shows the knob is controllable.

### Phase 6: Evaluation (the core result)

**Plain English:** replay every router against the saved test answers and see how much quality you keep for how much money.

For each router and test set (public, backend, combined), sweep τ (or *f* for random) across its full range. For each point, compute:
- **% calls to Sonnet**;
- **accuracy:** the Haiku answer's correctness if routed to Haiku, else Sonnet's;
- **cost:** the recorded per-prompt cost of the chosen model; report it as a % of all-Sonnet cost;
- **latency:** mean and p95 using recorded latencies where available; note the batch limitation.

**Metrics** (define them in the README in plain English):
- **PGR** (performance gap recovered) at a point = (router accuracy − Haiku accuracy) / (Sonnet accuracy − Haiku accuracy).
- **APGR:** average PGR across the sweep (area under the PGR vs %-Sonnet curve), as in RouteLLM.
- **CPT(x%):** the minimum % of calls to Sonnet needed to reach PGR = x%. Report CPT(50%) and CPT(80%).
- **Headline:** cost (% of all-Sonnet) at which the router keeps **≥95% of Sonnet's accuracy**, plus accuracy at a fixed 30% Sonnet-call rate.
- **AUROC** of each router's score for predicting Haiku correctness.
- **Uncertainty:** 1,000 bootstrap resamples of the test prompts, with 95% CIs for every headline metric.

**Figures** (`results/figures/`):
1. Cost-quality curves, all routers plus the random baseline, one per test set.
2. PGR vs % Sonnet calls.
3. Reliability diagrams (Phase 5).
4. DistilBERT train vs validation curves.
5. Per-category accuracy and routing share on the backend set.

### Phase 7: Honesty tests

1. **Generalization:** routers are trained on public GSM8K/MMLU. Compare their performance on the **public test** versus the **backend set**. Report how much APGR drops and which router degrades least.
2. **Simple vs complex:** does DistilBERT beat logistic regression on test, with CIs? If not, say so and give the likely reason (data size, overfitting).
3. **Per-category breakdown** (backend set and MMLU subjects): where routing saves the most, and where it fails (e.g. "looks easy but isn't" prompts like Python gotchas).
4. **Overfitting check:** the train vs validation vs test gap for each learned router.
5. **Sanity checks:** the random baseline's measured curve matches its expected curve; the oracle is the upper envelope; no router beats the oracle.

Write these up as a **"What worked / what didn't"** section.

### Phase 9: Write-up and outputs

- **README.md** (recruiter-skimmable at the top, detailed below):
  1. One-paragraph summary with the headline numbers (with CIs).
  2. The main cost-quality figure.
  3. Setup: models, data, routers, in plain English.
  4. Results table: all routers × test sets (APGR, CPT(50%), CPT(80%), cost at 95% of Sonnet's accuracy, AUROC, ECE).
  5. What worked / what didn't (Phase 7).
  6. Limitations: sample sizes, one model pair, benchmark answers may appear in model training data, batch latency caveat.
  7. Reproduce: exact commands per phase, expected cost.
  8. Total API spend, from the ledger.
- `results/summary.json`: machine-readable headline numbers.
- `results/RESUME_NOTES.md`: draft resume bullets (one long, one short) and a ~300-character paragraph for a job-application form, **using only numbers from `summary.json`**. Mark them as drafts for Sahil to edit.
- 🛑 **CHECKPOINT:** walk Sahil through the results in plain English, including what he should be ready to explain in an interview.

## 7. Definition of done

- [ ] All phases 0–7 and 9 complete; every 🛑 checkpoint confirmed by Sahil.
- [ ] Total API spend ≤ $5, shown in `results/spend_ledger.csv`.
- [ ] Re-running every script uses cached responses and costs $0.
- [ ] `pytest` passes, including tests for answer extraction/grading, the backend-set verifiers, metric functions (PGR/APGR/CPT on toy data), and no train/test overlap.
- [ ] Fixed seeds; the same commands reproduce the same numbers (DistilBERT within small noise).
- [ ] README numbers match `summary.json`; nothing is hand-typed or invented.
- [ ] No secrets in the repo.

## 8. Suggested commands (adjust names as you build)

```bash
python scripts/phase0_smoke.py
python scripts/phase1_train_data.py --estimate      # prints cost, no calls
python scripts/phase1_train_data.py --run
python scripts/phase2_build_backend_set.py          # free; verifies answers by execution
python scripts/phase2_test_data.py --estimate
python scripts/phase2_test_data.py --run
python scripts/phase3_embed.py
python scripts/phase4_train_routers.py               # includes DistilBERT on MPS
python scripts/phase5_calibrate.py
python scripts/phase6_evaluate.py
python scripts/phase7_honesty.py
python scripts/phase9_report.py                      # figures, tables, README sections, summary.json
```
