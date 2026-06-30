import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from config import DISTANCE_LEVELS, DISTRACTOR_COUNT_LEVELS
from score import aggregate, chance_accuracy, confidence_interval

_FIGURES_DIR = "figures"

_MODEL_DISPLAY = {
    "gpt-4o-mini": "GPT-4o-mini",
    "llama-3.1-8b": "Llama 3.1 8B",
    "mistral-7b": "Mistral 7B",
    "phi-3.5-mini": "Phi-3.5 Mini",
}

# Ordered strongest/largest reference → smallest local model.
_MODEL_ORDER = ["gpt-4o-mini", "llama-3.1-8b", "mistral-7b", "phi-3.5-mini"]

_MARKERS = ["o", "s", "^", "D"]

# Academic qualitative palette — high contrast, colourblind-safe (ColorBrewer-inspired).
# Navy, Crimson, Forest green, Amber.
_COLORS = [
    "#1B4F8A",  # navy
    "#C0392B",  # crimson
    "#1A7A4A",  # forest green
    "#D4870A",  # amber
]
_HEATMAP_CMAP = "Blues"


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


def _ordered_models_present(df: pd.DataFrame) -> list[str]:
    return [m for m in _MODEL_ORDER if m in set(df["model"])]


def _summarise_binary_accuracy(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Summarise per-document binary correctness into accuracy and normal-approximation
    binomial 95% confidence intervals. This matches the thesis definition:
    Acc ± 1.96 * sqrt(Acc(1 - Acc) / n).
    """
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_documents=("correct", "size"),
            n_correct=("correct", "sum"),
            mean_n_sentences=("n_sentences", "mean"),
        )
    )
    summary["accuracy"] = summary["n_correct"] / summary["n_documents"]

    ci_values = [
        confidence_interval(float(row.accuracy), int(row.n_documents))
        for row in summary.itertuples(index=False)
    ]
    summary["ci_low"] = [lo for lo, _ in ci_values]
    summary["ci_high"] = [hi for _, hi in ci_values]
    summary["chance_accuracy"] = summary["mean_n_sentences"].apply(chance_accuracy)
    return summary


# Figure 1: Accuracy vs. distance, collapsed across distractor levels


def _figure1(rows: list[dict]) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    raw_df = pd.DataFrame(rows)
    df = _summarise_binary_accuracy(raw_df, ["model", "distance"])
    models_present = _ordered_models_present(df)

    for model, marker, color in zip(
        models_present,
        _MARKERS[: len(models_present)],
        _COLORS[: len(models_present)],
    ):
        mdf = df[df["model"] == model].sort_values("distance")
        x = mdf["distance"].to_numpy(dtype=float)
        y = mdf["accuracy"].to_numpy(dtype=float)
        ci_low = mdf["ci_low"].to_numpy(dtype=float)
        ci_high = mdf["ci_high"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=6,
            label=_MODEL_DISPLAY.get(model, model),
        )
        ax.fill_between(
            x,
            ci_low,
            ci_high,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    # Chance varies slightly with document length; use the empirical mean over
    # evaluated documents as the displayed reference line.
    chance_level = raw_df["n_sentences"].apply(chance_accuracy).mean()
    ax.axhline(
        chance_level,
        color="#444444",
        linestyle=":",
        linewidth=1.5,
        label=f"Chance baseline ({chance_level:.4f})",
    )

    ax.set_xlim(DISTANCE_LEVELS[0] - 1, DISTANCE_LEVELS[-1] + 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(DISTANCE_LEVELS)
    ax.set_xlabel("Contradiction distance |j − i| (sentence indices)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. contradiction distance", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    _save(fig, "figure1_accuracy_vs_distance")
    plt.close(fig)


# Figure 2: Heatmap grid


def _figure2(agg_rows: list[dict]) -> None:
    sns.set_style("white")
    df = pd.DataFrame(agg_rows)

    models_present = _ordered_models_present(df)
    vmax = 1.0  # Fixed full range so colour scale is interpretable across models.

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax_idx, model in enumerate(models_present):
        ax = axes_flat[ax_idx]
        mdf = df[df["model"] == model]

        pivot = mdf.pivot_table(
            index="distractor_count",
            columns="distance",
            values="accuracy",
            aggfunc="mean",
        )
        pivot = pivot.reindex(
            index=DISTRACTOR_COUNT_LEVELS,
            columns=DISTANCE_LEVELS,
        )

        sns.heatmap(
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
        ax.set_xlabel("Distance |j − i|" if ax_idx >= 2 else "", fontsize=10)
        ax.set_ylabel("Distractor pairs" if ax_idx % 2 == 0 else "", fontsize=10)

    for ax_idx in range(len(models_present), 4):
        axes_flat[ax_idx].set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap=_HEATMAP_CMAP, norm=plt.Normalize(vmin=0.0, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat, fraction=0.03, pad=0.04)
    cbar.set_label("Mean accuracy", fontsize=11)

    fig.suptitle(
        "Accuracy across distance × distractor density grid",
        fontsize=13,
    )

    _save(fig, "figure2_heatmap_grid")
    plt.close(fig)


# Figure 3: Degradation slopes, using WLS over per-distance accuracies


def _wls_slope_with_ci(x, y, weights) -> tuple[float, float, float]:
    """
    Weighted least-squares slope and 95% CI for y ~ 1 + x.
    The input y values are per-distance accuracies; weights are the number of
    evaluated documents contributing to each accuracy estimate.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    if len(x) < 3:
        raise ValueError("Need at least three distance levels to estimate a slope CI.")

    x_design = np.column_stack([np.ones_like(x), x])
    xtw = x_design.T * w
    xtwx_inv = np.linalg.inv(xtw @ x_design)
    beta = xtwx_inv @ (xtw @ y)
    fitted = x_design @ beta
    resid = y - fitted

    df_resid = len(x) - x_design.shape[1]
    sigma2 = float((w * resid**2).sum() / df_resid)
    cov_beta = sigma2 * xtwx_inv
    slope = float(beta[1])
    slope_se = float(cov_beta[1, 1] ** 0.5)
    t_crit = float(stats.t.ppf(0.975, df=df_resid))

    return slope, slope - t_crit * slope_se, slope + t_crit * slope_se


def _figure3(rows: list[dict]) -> None:
    sns.set_style("whitegrid")
    raw_df = pd.DataFrame(rows)
    df = _summarise_binary_accuracy(raw_df, ["model", "distance"])

    models_present = _ordered_models_present(df)

    slopes, ci_lows, ci_highs, labels = [], [], [], []

    for model in models_present:
        mdf = df[df["model"] == model].sort_values("distance")
        slope, ci_lo, ci_hi = _wls_slope_with_ci(
            x=mdf["distance"].to_numpy(dtype=float),
            y=mdf["accuracy"].to_numpy(dtype=float),
            weights=mdf["n_documents"].to_numpy(dtype=float),
        )

        slopes.append(slope)
        ci_lows.append(ci_lo)
        ci_highs.append(ci_hi)
        labels.append(_MODEL_DISPLAY.get(model, model))

    # Reverse so the top of the plot follows _MODEL_ORDER.
    slopes_rev = list(reversed(slopes))
    ci_lows_rev = list(reversed(ci_lows))
    ci_highs_rev = list(reversed(ci_highs))
    labels_rev = list(reversed(labels))
    colors_rev = list(reversed(_COLORS[: len(labels)]))

    y_pos = list(range(len(labels_rev)))

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels_rev) * 1.2)))

    for i, (slope, lo, hi, color) in enumerate(
        zip(slopes_rev, ci_lows_rev, ci_highs_rev, colors_rev)
    ):
        ax.errorbar(
            slope,
            i,
            xerr=[[slope - lo], [hi - slope]],
            fmt="o",
            color=color,
            markersize=8,
            linewidth=1.8,
            capsize=5,
            capthick=1.5,
        )

    ax.axvline(0.0, linestyle="--", color="#444444", linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_rev, fontsize=11)
    ax.set_xlabel("Accuracy change per sentence-index distance", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Distance-degradation slope by model", fontsize=13)

    _save(fig, "figure3_degradation_slopes")
    plt.close(fig)


# Public entry point


def make_all_figures(results_paths: list[str]) -> None:
    rows = _load_rows(results_paths)
    agg_rows = aggregate(rows)

    print("Generating Figure 1 …")
    _figure1(rows)

    print("Generating Figure 2 …")
    _figure2(agg_rows)

    print("Generating Figure 3 …")
    _figure3(rows)

    print("All figures saved to figures/")
