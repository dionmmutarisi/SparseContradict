import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import DISTANCE_LEVELS, DISTRACTOR_COUNT_LEVELS, MODELS
from score import aggregate, chance_accuracy

_FIGURES_DIR = "figures"

_MODEL_DISPLAY = {
    "gpt-4o-mini": "GPT-4o-mini",
    "llama-3.1-8b": "Llama 3.1 8B",
    "mistral-7b": "Mistral 7B",
    "phi-3.5-mini": "Phi-3.5 Mini",
}

# Ordered largest → smallest parameter count for Figure 3
_MODEL_ORDER = ["gpt-4o-mini", "llama-3.1-8b", "mistral-7b", "phi-3.5-mini"]

_MARKERS = ["o", "s", "^", "D"]
# Muted, publication-friendly palette with strong contrast for model lines.
_COLORS = [
    "#2E5D7F",  # slate blue
    "#B25D5D",  # muted terracotta
    "#5B8C6E",  # muted sage
    "#7B5E8C",  # muted plum
]
_HEATMAP_CMAP = "YlGnBu"


def _load_rows(results_paths: list[str]) -> list[dict]:
    rows = []
    for path in results_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _save(fig, name: str) -> None:
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    base = os.path.join(_FIGURES_DIR, name)
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    print(f"  Saved {base}.pdf and {base}.png")


# ---------------------------------------------------------------------------
# Figure 1: Accuracy vs. distance (collapsed across distractors)
# ---------------------------------------------------------------------------

def _figure1(agg_rows: list[dict]) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    df = pd.DataFrame(agg_rows)

    # Mean chance accuracy across all documents
    mean_chance = df["chance_accuracy"].mean()

    models_present = [m for m in _MODEL_ORDER if m in df["model"].values]

    for model, marker, color in zip(
        models_present,
        _MARKERS[: len(models_present)],
        _COLORS[: len(models_present)],
    ):
        mdf = (
            df[df["model"] == model]
            .groupby("distance", as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                ci_low=("ci_low", "mean"),
                ci_high=("ci_high", "mean"),
            )
            .sort_values("distance")
        )
        distances = mdf["distance"].values
        accs = mdf["accuracy"].values
        lo = mdf["ci_low"].values
        hi = mdf["ci_high"].values

        ax.plot(
            distances, accs,
            marker=marker, color=color, linewidth=1.8, markersize=6,
            label=_MODEL_DISPLAY.get(model, model),
        )
        ax.fill_between(distances, lo, hi, alpha=0.15, color=color)

    ax.axhline(
        mean_chance, linestyle="--", color="black", linewidth=1.2,
        label=f"chance (≈ {mean_chance:.4f})",
    )

    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(DISTANCE_LEVELS)
    ax.set_xlabel("Contradiction distance (sentences)", fontsize=12)
    ax.set_ylabel("Mean accuracy", fontsize=12)
    ax.set_title("Figure 1: Accuracy vs. contradiction distance", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)

    _save(fig, "figure1_accuracy_vs_distance")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Heatmap grid
# ---------------------------------------------------------------------------

def _figure2(agg_rows: list[dict]) -> None:
    sns.set_style("white")
    df = pd.DataFrame(agg_rows)

    models_present = [m for m in _MODEL_ORDER if m in df["model"].values]
    vmax = df["accuracy"].max()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes_flat = axes.flatten()

    # Shared colorbar image reference
    img = None

    for ax_idx, model in enumerate(models_present):
        ax = axes_flat[ax_idx]
        mdf = df[df["model"] == model]

        pivot = mdf.pivot_table(
            index="distractor_count",
            columns="distance",
            values="accuracy",
            aggfunc="mean",
        )
        # Ensure canonical ordering
        pivot = pivot.reindex(
            index=DISTRACTOR_COUNT_LEVELS,
            columns=DISTANCE_LEVELS,
        )

        img = sns.heatmap(
            pivot,
            ax=ax,
            vmin=0.0,
            vmax=vmax,
            cmap=_HEATMAP_CMAP,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 9},
            linewidths=0.5,
            cbar=False,
        )
        ax.set_title(_MODEL_DISPLAY.get(model, model), fontsize=11)
        ax.set_xlabel("Distance" if ax_idx >= 2 else "", fontsize=10)
        ax.set_ylabel("Distractors" if ax_idx % 2 == 0 else "", fontsize=10)

    # Hide any unused subplots
    for ax_idx in range(len(models_present), 4):
        axes_flat[ax_idx].set_visible(False)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=_HEATMAP_CMAP, norm=plt.Normalize(vmin=0.0, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat, fraction=0.03, pad=0.04)
    cbar.set_label("Mean accuracy", fontsize=11)

    fig.suptitle(
        "Figure 2: Accuracy across distance × distractor density grid",
        fontsize=13,
    )

    _save(fig, "figure2_heatmap_grid")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Degradation slopes (OLS coefficient plot)
# ---------------------------------------------------------------------------

def _figure3(agg_rows: list[dict]) -> None:
    sns.set_style("whitegrid")
    df = pd.DataFrame(agg_rows)

    models_present = [m for m in _MODEL_ORDER if m in df["model"].values]

    slopes, ci_lows, ci_highs, labels = [], [], [], []

    for model in models_present:
        mdf = (
            df[df["model"] == model]
            .groupby("distance", as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                n_documents=("n_documents", "sum"),
            )
        )
        x = mdf["distance"].values.astype(float)
        y = mdf["accuracy"].values
        w = mdf["n_documents"].values.astype(float)

        # Weighted OLS via WLS
        result = stats.linregress(x, y)
        slope = result.slope
        # 95% CI from standard error
        t_crit = stats.t.ppf(0.975, df=len(x) - 2)
        se = result.stderr
        ci_lo = slope - t_crit * se
        ci_hi = slope + t_crit * se

        slopes.append(slope)
        ci_lows.append(ci_lo)
        ci_highs.append(ci_hi)
        labels.append(_MODEL_DISPLAY.get(model, model))

    # Reverse so top of plot = largest model (GPT-4o-mini)
    y_pos = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 1.2)))

    for i, (slope, lo, hi, label) in enumerate(
        zip(slopes, ci_lows, ci_highs, labels)
    ):
        ax.errorbar(
            slope,
            i,
            xerr=[[slope - lo], [hi - slope]],
            fmt="o",
            color=_COLORS[i % len(_COLORS)],
            markersize=8,
            linewidth=1.8,
            capsize=5,
            capthick=1.5,
        )

    ax.axvline(0.0, linestyle="--", color="black", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Accuracy change per sentence of distance", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Figure 3: Degradation slope by model", fontsize=13)

    _save(fig, "figure3_degradation_slopes")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def make_all_figures(results_paths: list[str]) -> None:
    rows = _load_rows(results_paths)
    agg_rows = aggregate(rows)

    print("Generating Figure 1 …")
    _figure1(agg_rows)

    print("Generating Figure 2 …")
    _figure2(agg_rows)

    print("Generating Figure 3 …")
    _figure3(agg_rows)

    print("All figures saved to figures/")
