"""Honesty tests: generalization, simple-vs-complex, per-category breakdown,
overfitting, and sanity checks. Writes results/tables/honesty_*.csv and
results/honesty.json.

    uv run python scripts/phase7_honesty.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dispatch_router import plots
from dispatch_router.config import path
from dispatch_router.evaluate import (
    apgr, curve, oracle_order, order_for, paired_bootstrap_diff, random_curve,
)
from dispatch_router.scoring import CAL_SCORES, LEARNED, subsets, test_frame, train_pool, with_router_scores

N_BOOT = 1000


def main() -> None:
    res_dir, tables = path("results_dir"), path("results_dir") / "tables"
    res = pd.read_csv(tables / "results_all.csv")
    cal = pd.read_csv(CAL_SCORES)
    routers = [r for r in LEARNED if r in set(cal.router)]
    df = with_router_scores(test_frame(), cal, routers)
    out: dict = {}

    # 1. Generalization: public -> backend
    gen = []
    for r in routers:
        p = res[(res.set == "public") & (res.router == r)].iloc[0]
        b = res[(res.set == "backend") & (res.router == r)].iloc[0]
        gen.append({"router": r, "apgr_public": p.apgr, "apgr_public_ci": [p.apgr_lo, p.apgr_hi],
                    "apgr_backend": b.apgr, "apgr_backend_ci": [b.apgr_lo, b.apgr_hi],
                    "apgr_drop": p.apgr - b.apgr, "auroc_public": p.auroc, "auroc_backend": b.auroc})
    out["generalization"] = gen
    pd.DataFrame(gen).to_csv(tables / "honesty_generalization.csv", index=False)

    # 2. Simple vs complex: paired bootstrap of DistilBERT - logreg (and kNN - logreg)
    cmp_rows = []
    for other in [r for r in ("distilbert", "knn", "prior") if r in routers]:
        for set_name, d in subsets(df).items():
            if set_name not in ("public", "backend", "combined"):
                continue
            args = (d.correct_h.astype(int).values, d.correct_s.astype(int).values,
                    d.cost_usd_h.values, d.cost_usd_s.values, d[other].values, d["logreg"].values)
            for key in ("apgr", "auroc"):
                pt, ci = paired_bootstrap_diff(*args, key=key, n_boot=N_BOOT, seed=2)
                cmp_rows.append({"comparison": f"{other} - logreg", "set": set_name, "metric": key,
                                 "diff": pt, "ci_lo": ci[0], "ci_hi": ci[1],
                                 "significant": not (ci[0] <= 0 <= ci[1])})
    out["vs_logreg"] = cmp_rows
    pd.DataFrame(cmp_rows).to_csv(tables / "honesty_vs_logreg.csv", index=False)

    # 3. Per-category breakdown (backend categories x tier, and MMLU subjects)
    be = df[df.set == "backend"].copy()
    k30 = int(round(0.3 * len(be)))
    to_sonnet = np.zeros(len(be), bool)
    to_sonnet[order_for(be.logreg.values)[:k30]] = True
    be["to_sonnet_30"] = to_sonnet
    cat = be.groupby(["category", "tier"]).agg(
        n=("id", "size"), acc_haiku=("correct_h", "mean"), acc_sonnet=("correct_s", "mean"),
        n_only_sonnet_right=("id", lambda i: int((~be.loc[i.index, "correct_h"] & be.loc[i.index, "correct_s"]).sum())),
        sonnet_share_at_30=("to_sonnet_30", "mean"),
        caught=("id", lambda i: int((be.loc[i.index, "to_sonnet_30"] & ~be.loc[i.index, "correct_h"]
                                     & be.loc[i.index, "correct_s"]).sum())),
    ).reset_index()
    cat.to_csv(tables / "honesty_backend_categories.csv", index=False)
    out["backend_categories"] = cat.to_dict("records")

    bt = be.groupby("category").agg(acc_haiku=("correct_h", "mean"), acc_sonnet=("correct_s", "mean"),
                                    sonnet_share=("to_sonnet_30", "mean")) * 100
    plots.backend_breakdown(bt, res_dir / "figures" / "backend_categories.png")

    mm = df[df.dataset == "mmlu"].copy()
    k30 = int(round(0.3 * len(mm)))
    ts = np.zeros(len(mm), bool)
    ts[order_for(mm.logreg.values)[:k30]] = True
    mm["to_sonnet_30"] = ts
    subj = mm.groupby("category").agg(
        n=("id", "size"), acc_haiku=("correct_h", "mean"), acc_sonnet=("correct_s", "mean"),
        sonnet_share_at_30=("to_sonnet_30", "mean")).reset_index()
    subj["gap"] = subj.acc_sonnet - subj.acc_haiku
    subj.sort_values("gap", ascending=False).to_csv(tables / "honesty_mmlu_subjects.csv", index=False)
    big = subj[subj.n >= 20]
    out["mmlu_subjects_largest_gap"] = big.nlargest(5, "gap").to_dict("records")
    out["mmlu_routing_vs_gap_spearman"] = float(big[["gap", "sonnet_share_at_30"]].corr("spearman").iloc[0, 1])

    # 4. Overfitting: AUROC on train / val / test sets
    pool = train_pool().set_index("id").correct
    labels = pd.concat([pool, df.set_index("id").correct_h]).astype(int)
    of = []
    for r in routers:
        d = cal[cal.router == r]
        row = {"router": r}
        for s in ("train", "val", "test_public", "backend"):
            x = d[d.set == s]
            row[f"auroc_{s}"] = roc_auc_score(labels[x.id], x.score) if len(x) else np.nan
        row["train_minus_val"] = row["auroc_train"] - row["auroc_val"]
        row["val_minus_test_public"] = row["auroc_val"] - row["auroc_test_public"]
        of.append(row)
    out["overfitting"] = of
    pd.DataFrame(of).to_csv(tables / "honesty_overfitting.csv", index=False)

    # 5. Sanity checks
    sanity = []
    rng = np.random.default_rng(3)
    for set_name, d in subsets(df).items():
        h, s = d.correct_h.astype(int).values, d.correct_s.astype(int).values
        ch, cs = d.cost_usd_h.values, d.cost_usd_s.values
        measured = np.mean([apgr(curve(h, s, ch, cs, rng.permutation(len(h)))) for _ in range(200)])
        orc = curve(h, s, ch, cs, oracle_order(h, s))
        beats = {r: bool(np.any(curve(h, s, ch, cs, order_for(d[r].values)).accuracy > orc.accuracy + 1e-12))
                 for r in routers}
        sanity.append({"set": set_name, "random_apgr_expected": apgr(random_curve(h, s, ch, cs)),
                       "random_apgr_measured_mean_200": measured,
                       "any_router_beats_oracle": any(beats.values())})
    out["sanity"] = sanity
    pd.DataFrame(sanity).to_csv(tables / "honesty_sanity.csv", index=False)

    (res_dir / "honesty.json").write_text(json.dumps(out, indent=2, default=float))
    for k in ("generalization", "vs_logreg", "overfitting", "sanity"):
        print(f"\n== {k}\n" + pd.DataFrame(out[k]).round(3).to_string(index=False))
    print("\n== backend categories (logreg, 30% to Sonnet)\n" + cat.round(3).to_string(index=False))
    print("\n== MMLU subjects with largest Sonnet-Haiku gap (n>=20)\n"
          + big.nlargest(5, "gap").round(3).to_string(index=False))
    print(f"\nSpearman(subject gap, share routed to Sonnet) = {out['mmlu_routing_vs_gap_spearman']:.3f}")


if __name__ == "__main__":
    main()
