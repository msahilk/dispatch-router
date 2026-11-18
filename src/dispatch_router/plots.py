"""Matplotlib figures for the README (static PNGs, light surface).

Router colours follow a fixed categorical order (validated for colour-vision
deficiency); reference curves are neutral grey. Every chart has a legend and the
same numbers are in results/tables/, so colour is never the only key.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#2b2b29"
MUTED = "#6f6e69"
GRID = "#e4e3df"
COLORS = {"logreg": "#2a78d6", "distilbert": "#eb6834", "knn": "#1baf7a", "prior": "#eda100"}
LABELS = {"logreg": "Logistic regression", "distilbert": "DistilBERT", "knn": "kNN",
          "prior": "Subject prior (no NLP)", "random": "Random (expected)", "oracle": "Oracle (upper bound)"}
REF_STYLE = {"random": dict(color="#8d8c86", ls="--", lw=1.5), "oracle": dict(color="#4a4945", ls=":", lw=2)}


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _fig(n=1, w=6.4, h=4.4):
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), facecolor=SURFACE, squeeze=False)
    return fig, axes[0]


def _save(fig, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(file, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _plot_curves(ax, curves: dict, x: str, y: str):
    for name in ("oracle", "random"):
        if name in curves:
            c = curves[name]
            ax.plot(getattr(c, x), getattr(c, y), label=LABELS[name], **REF_STYLE[name])
    for name, col in COLORS.items():
        if name in curves:
            c = curves[name]
            ax.plot(getattr(c, x), getattr(c, y), color=col, lw=2, label=LABELS[name])


def cost_quality(curves_by_set: dict[str, dict], file: Path, titles: dict[str, str]):
    """Accuracy vs cost (% of all-Sonnet), one panel per test set."""
    fig, axes = _fig(len(curves_by_set))
    for ax, (set_name, curves) in zip(axes, curves_by_set.items()):
        any_c = next(iter(curves.values()))
        x = lambda c: c.cost_pct  # noqa: E731
        for name in ("oracle", "random"):
            if name in curves:
                ax.plot(x(curves[name]), 100 * curves[name].accuracy, label=LABELS[name], **REF_STYLE[name])
        for name, col in COLORS.items():
            if name in curves:
                ax.plot(x(curves[name]), 100 * curves[name].accuracy, color=col, lw=2, label=LABELS[name])
        h0, s0 = 100 * any_c.acc_haiku, 100 * any_c.acc_sonnet
        ax.axhline(s0, color=GRID, lw=1)
        ax.annotate(f"all-Sonnet {s0:.1f}%", (100, s0), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=MUTED)
        ax.annotate(f"all-Haiku {h0:.1f}%", (x(any_c)[0], h0), xytext=(4, -10), textcoords="offset points",
                    fontsize=8, color=MUTED)
        _style(ax, titles.get(set_name, set_name), "Cost (% of always-Sonnet)", "Accuracy (%)")
        ax.set_xlim(0, 102)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, file)


def pgr_curves(curves_by_set: dict[str, dict], file: Path, titles: dict[str, str]):
    fig, axes = _fig(len(curves_by_set))
    for ax, (set_name, curves) in zip(axes, curves_by_set.items()):
        for name in ("oracle", "random"):
            if name in curves:
                ax.plot(100 * curves[name].frac_sonnet, 100 * curves[name].pgr, label=LABELS[name], **REF_STYLE[name])
        for name, col in COLORS.items():
            if name in curves:
                ax.plot(100 * curves[name].frac_sonnet, 100 * curves[name].pgr, color=col, lw=2, label=LABELS[name])
        for lvl in (50, 80):
            ax.axhline(lvl, color=GRID, lw=1)
        _style(ax, titles.get(set_name, set_name), "Calls routed to Sonnet (%)", "Performance gap recovered (%)")
        ax.set_xlim(0, 100)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, file)


def reliability(panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], file: Path):
    """panels: (title, mp_raw, fp_raw, n_raw, mp_cal, fp_cal, n_cal). Bins with fewer
    than MIN_BIN prompts are hidden: one or two prompts make a 0% or 100% point
    that looks dramatic and means nothing."""
    MIN_BIN = 10
    fig, axes = _fig(len(panels), w=4.2, h=4.2)
    for ax, (title, mp_r, fp_r, n_r, mp_c, fp_c, n_c) in zip(axes, panels):
        mp_r, fp_r = mp_r[n_r >= MIN_BIN], fp_r[n_r >= MIN_BIN]
        mp_c, fp_c = mp_c[n_c >= MIN_BIN], fp_c[n_c >= MIN_BIN]
        ax.plot([0, 1], [0, 1], color="#8d8c86", ls="--", lw=1, label="Perfectly calibrated")
        ax.plot(mp_r, fp_r, "o-", color="#b4b2a9", lw=1.5, ms=5, label="Before (raw)")
        ax.plot(mp_c, fp_c, "o-", color=COLORS["logreg"], lw=2, ms=6, label="After (Platt)")
        _style(ax, title, "Predicted p(Haiku correct)", "Actual Haiku accuracy")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.text(0.98, 0.03, f"bins with < {MIN_BIN} prompts hidden", ha="right", fontsize=7, color=MUTED,
                transform=ax.transAxes)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, file)


def train_val_curves(history, file: Path):
    ep = [h["epoch"] for h in history]
    fig, axes = _fig(2, w=4.8, h=3.8)
    for ax, key, title in ((axes[0], "auroc", "AUROC (higher is better)"), (axes[1], "logloss", "Log-loss")):
        ax.plot(ep, [h[f"train_{key}"] for h in history], "o-", color=COLORS["distilbert"], lw=2, label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], "s--", color=COLORS["logreg"], lw=2, label="validation")
        _style(ax, f"DistilBERT {title}", "Epoch", key)
        ax.set_xticks(ep)
    axes[0].legend(frameon=False, fontsize=9)
    _save(fig, file)


def backend_breakdown(table, file: Path):
    """table: index=category, columns acc_haiku, acc_sonnet, sonnet_share (all in %)."""
    cats = list(table.index)
    y = np.arange(len(cats))
    fig, axes = _fig(2, w=5.2, h=0.6 * len(cats) + 1.6)
    ax = axes[0]
    ax.barh(y - 0.2, table.acc_haiku, height=0.36, color="#86b6ef", label="Haiku", edgecolor=SURFACE, lw=2)
    ax.barh(y + 0.2, table.acc_sonnet, height=0.36, color=COLORS["logreg"], label="Sonnet", edgecolor=SURFACE, lw=2)
    for i, (a, b) in enumerate(zip(table.acc_haiku, table.acc_sonnet)):
        ax.text(a + 0.5, i - 0.2, f"{a:.0f}", va="center", fontsize=8, color=INK)
        ax.text(b + 0.5, i + 0.2, f"{b:.0f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, cats)
    _style(ax, "Accuracy by category", "Accuracy (%)", "")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    ax = axes[1]
    ax.barh(y, table.sonnet_share, height=0.5, color=COLORS["logreg"], edgecolor=SURFACE, lw=2)
    for i, v in enumerate(table.sonnet_share):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, cats)
    _style(ax, "Share routed to Sonnet (logreg, 30% budget)", "% of prompts", "")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    _save(fig, file)
