"""Calibrate each learned router on the validation split (Platt scaling) and report
Brier score / ECE before and after, reliability diagrams, and threshold control.

    uv run python scripts/phase5_calibrate.py
"""

import json

import numpy as np
import pandas as pd

from dispatch_router import plots
from dispatch_router.calibrate import Platt, brier, ece, realized_rate, reliability, threshold_for_rate
from dispatch_router.config import path
from dispatch_router.scoring import CAL_SCORES, LEARNED, RAW_SCORES, prior_scores, test_frame, train_pool

TARGETS = [0.10, 0.20, 0.30, 0.50]


def main() -> None:
    raw = pd.read_csv(RAW_SCORES)
    pool = train_pool()
    train = pool[pool.split == "train"]
    test = test_frame()

    # Labels for every scored prompt: train/val from Haiku's training labels, test from test_haiku
    labels = pd.concat([pool[["id", "correct"]], test[["id"]].assign(correct=test.correct_h)])
    labels = labels.drop_duplicates("id").set_index("id").correct.astype(int)

    # Add the no-NLP prior baseline as a 'router'
    meta = pd.concat([pool[["id", "dataset", "category", "split"]],
                      test[["id", "dataset", "category", "set"]].rename(columns={"set": "split"})])
    prior = pd.DataFrame({"id": meta.id, "set": meta.split, "router": "prior",
                          "score": prior_scores(train, meta)})
    raw = pd.concat([raw[raw.router != "prior"], prior], ignore_index=True)

    tables = path("results_dir") / "tables"
    rows, thr_rows, out, panels, params = [], [], [], [], {}
    for r in LEARNED:
        d = raw[raw.router == r]
        if d.empty:
            print(f"(skipping {r}: no scores)")
            continue
        val = d[d.set == "val"]
        cal = Platt().fit(val.score.values, labels[val.id].values)
        params[r] = cal.params
        d = d.assign(score_cal=cal.transform(d.score.values))
        out.append(d)
        for s in ("val", "test_public", "backend"):
            x = d[d.set == s]
            y = labels[x.id].values
            rows.append({"router": r, "set": s, "n": len(x),
                         "brier_raw": brier(x.score, y), "brier_cal": brier(x.score_cal, y),
                         "ece_raw": ece(x.score, y), "ece_cal": ece(x.score_cal, y)})
        # threshold control: choose tau on validation, measure realized Sonnet-call rate on test
        vs = d[d.set == "val"].score_cal.values
        for t in TARGETS:
            tau = threshold_for_rate(vs, t)
            thr_rows.append({"router": r, "target_sonnet_rate": t, "tau": tau,
                             **{f"realized_{s}": realized_rate(d[d.set == s].score_cal, tau)
                                for s in ("val", "test_public", "backend")}})
        x = d[d.set == "test_public"]
        y = labels[x.id].values
        mr, fr, nr = reliability(x.score, y)
        mc, fc, nc = reliability(x.score_cal, y)
        panels.append((plots.LABELS[r], mr, fr, nr, mc, fc, nc))

    pd.concat(out).to_csv(CAL_SCORES, index=False)
    cal_df = pd.DataFrame(rows).round(4)
    thr_df = pd.DataFrame(thr_rows).round(4)
    cal_df.to_csv(tables / "calibration.csv", index=False)
    thr_df.to_csv(tables / "threshold_control.csv", index=False)
    (tables / "platt_params.json").write_text(json.dumps(params, indent=2))
    plots.reliability(panels, path("results_dir") / "figures" / "reliability_public_test.png")

    print("Calibration (fit on validation; Platt scaling)\n")
    print(cal_df.to_string(index=False))
    print("\nThreshold control: tau chosen on validation for a target Sonnet-call rate\n")
    print(thr_df.to_string(index=False))


if __name__ == "__main__":
    main()
