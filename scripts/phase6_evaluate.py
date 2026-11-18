"""Replay every router against the recorded test answers: cost-quality curves,
PGR/APGR/CPT, AUROC, and 1,000-resample bootstrap CIs.

    uv run python scripts/phase6_evaluate.py
"""

import json

import numpy as np
import pandas as pd

from dispatch_router import plots
from dispatch_router.config import path
from dispatch_router.evaluate import (
    bootstrap, curve, metrics, oracle_order, order_for, random_curve, router_metrics,
)
from dispatch_router.scoring import CAL_SCORES, LEARNED, subsets, test_frame, with_router_scores

N_BOOT = 1000
TITLES = {"public": "Public test (GSM8K + MMLU)", "gsm8k": "GSM8K test", "mmlu": "MMLU test",
          "backend": "Backend set (all)", "backend_stress": "Backend set (stress tier)",
          "combined": "All test prompts"}


def main() -> None:
    df = test_frame()
    cal = pd.read_csv(CAL_SCORES)
    routers = [r for r in LEARNED if r in set(cal.router)]
    df = with_router_scores(df, cal, routers)

    res_dir = path("results_dir")
    rows, curves_by_set = [], {}
    for set_name, d in subsets(df).items():
        h, s = d.correct_h.astype(int).values, d.correct_s.astype(int).values
        ch, cs = d.cost_usd_h.values, d.cost_usd_s.values
        curves = {"random": random_curve(h, s, ch, cs), "oracle": curve(h, s, ch, cs, oracle_order(h, s))}
        fixed = {
            "always_haiku": {"cost_pct": 100 * ch.sum() / cs.sum(), "accuracy": h.mean()},
            "always_sonnet": {"cost_pct": 100.0, "accuracy": s.mean()},
        }
        base = {"set": set_name, "n": len(d), "acc_haiku": h.mean(), "acc_sonnet": s.mean(),
                "haiku_cost_pct": fixed["always_haiku"]["cost_pct"],
                "n_only_sonnet_right": int(((h == 0) & (s == 1)).sum()),
                "n_only_haiku_right": int(((h == 1) & (s == 0)).sum())}
        for name in ("random", "oracle"):
            rows.append({**base, "router": name, **metrics(curves[name])})
        for r in routers:
            score = d[r].values
            curves[r] = curve(h, s, ch, cs, order_for(score))
            m = router_metrics(h, s, ch, cs, score)
            ci = bootstrap(h, s, ch, cs, score, n_boot=N_BOOT, seed=1)
            rows.append({**base, "router": r, **m, **{f"{k}_lo": v[0] for k, v in ci.items()},
                         **{f"{k}_hi": v[1] for k, v in ci.items()}})
        curves_by_set[set_name] = curves
        print(f"{set_name:<15} n={len(d):<5} Haiku {h.mean():.3f}  Sonnet {s.mean():.3f}  "
              f"only-Sonnet-right {base['n_only_sonnet_right']}")

    res = pd.DataFrame(rows)
    res.to_csv(res_dir / "tables" / "results_all.csv", index=False)

    show = ["set", "router", "apgr", "cpt50", "cpt80", "cost_pct_at_95", "cost_pct_at_99", "cost_pct_at_100",
            "acc_at_30", "auroc"]
    print("\n" + res[show].round(3).to_string(index=False))

    # curves for later scripts (figures, honesty checks)
    np.savez_compressed(res_dir / "tables" / "curves.npz", **{
        f"{s}|{r}|{k}": getattr(c, k) for s, cs_ in curves_by_set.items() for r, c in cs_.items()
        for k in ("frac_sonnet", "accuracy", "cost_pct", "pgr")})

    figs = res_dir / "figures"
    plots.cost_quality({k: curves_by_set[k] for k in ("public", "mmlu", "backend")},
                       figs / "cost_quality.png", TITLES)
    plots.pgr_curves({k: curves_by_set[k] for k in ("public", "mmlu", "backend")}, figs / "pgr.png", TITLES)
    plots.cost_quality({k: curves_by_set[k] for k in ("gsm8k", "backend_stress", "combined")},
                       figs / "cost_quality_more.png", TITLES)
    print(f"\nfigures -> {figs}")


if __name__ == "__main__":
    main()
