"""
Model performance comparison visualisations.

Loads trained model metadata to compare Brier scores across all four
models with bar charts, a feature-count vs. performance scatter, and
a CSV performance summary table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# --- Constants -------------------------------------------------------------

_MODEL_ORDER = ["basic_model", "situation_model", "shottype_model", "advanced_model"]

_MODEL_LABELS = {
    "basic_model": "Basic\n(X, Y, dist, angle)",
    "situation_model": "Situation\n(+ situation one-hot)",
    "shottype_model": "Shot Type\n(+ shotType one-hot)",
    "advanced_model": "Advanced\n(+ interactions)",
}

# Seaborn colorblind palette colours
_COLOURS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

# Naïve baseline: constant predictor set to marginal goal rate.
# At ~10.5 % goal rate: Brier = p(1-p) = 0.105 * 0.895 ≈ 0.094
_NAIVE_BRIER = 0.094


# --- Data loading ----------------------------------------------------------

def load_all_model_metadata(models_dir: Path) -> list[dict]:
    """
    Loads ``metadata.json`` for every model found under ``models_dir``,
    ordered by ``_MODEL_ORDER`` where possible.

    Args:
        models_dir: Root directory containing one subdirectory per model.

    Returns:
        List of metadata dicts, each augmented with an ``n_features`` key.
    """
    results = []
    for model_name in _MODEL_ORDER:
        path = models_dir / model_name / "metadata.json"
        if not path.exists():
            print(f"  Warning: metadata not found for '{model_name}'.")
            continue
        with open(path, "r") as fh:
            meta = json.load(fh)
        meta["n_features"] = len(meta.get("features", []))
        results.append(meta)
    return results


# --- Visualisations --------------------------------------------------------

def plot_brier_score_comparison(metadata_list: list[dict], output_dir: Path) -> None:
    """
    Renders and saves a vertical bar chart comparing Brier scores across all
    models, with a naïve-baseline reference line and incremental improvement
    annotations.

    Args:
        metadata_list: List of model metadata dicts (from load_all_model_metadata).
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [_MODEL_LABELS.get(m["model_name"], m["model_name"]) for m in metadata_list]
    scores = [m["brier_score"] for m in metadata_list]
    colours = _COLOURS[: len(metadata_list)]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        names,
        scores,
        color=colours,
        edgecolor="white",
        linewidth=0.8,
        width=0.55,
        zorder=3,
    )

    # Value label above each bar
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.0003,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Incremental improvement arrows between adjacent bars
    for i in range(1, len(scores)):
        delta = scores[i - 1] - scores[i]
        if delta > 0:
            mid_x = i - 0.5
            ax.annotate(
                f"−{delta:.4f}",
                xy=(mid_x, scores[i] + 0.0005),
                fontsize=8,
                ha="center",
                color="#444",
                style="italic",
            )

    # Naïve baseline reference line
    ax.axhline(
        _NAIVE_BRIER,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Naïve baseline  ({_NAIVE_BRIER:.3f})",
        zorder=2,
    )

    ax.set_ylim(0.075, 0.098)
    ax.set_ylabel("Brier Score  (lower is better)", fontsize=11)
    ax.set_title("Model Performance Comparison — Brier Score", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = output_dir / "brier_score_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_brier_vs_feature_count(metadata_list: list[dict], output_dir: Path) -> None:
    """
    Saves a scatter plot of Brier score against feature set size to illustrate
    diminishing returns from additional features.

    Args:
        metadata_list: List of model metadata dicts (must contain ``n_features``).
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [_MODEL_LABELS.get(m["model_name"], m["model_name"]) for m in metadata_list]
    feature_counts = [m["n_features"] for m in metadata_list]
    scores = [m["brier_score"] for m in metadata_list]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Connecting line (sorted by feature count)
    order = np.argsort(feature_counts)
    ax.plot(
        [feature_counts[i] for i in order],
        [scores[i] for i in order],
        color="#888",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )

    for i, (n, s, label, colour) in enumerate(
        zip(feature_counts, scores, names, _COLOURS)
    ):
        ax.scatter(n, s, s=130, color=colour, zorder=5,
                   edgecolors="white", linewidths=1.5)
        v_offset = 0.00018 if i % 2 == 0 else -0.00028
        ax.annotate(
            label,
            xy=(n, s),
            xytext=(n + 0.6, s + v_offset),
            fontsize=8.5,
            color=colour,
            fontweight="bold",
        )

    # Naïve baseline
    ax.axhline(
        _NAIVE_BRIER,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"Naïve baseline  ({_NAIVE_BRIER:.3f})",
        zorder=3,
    )

    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("Brier Score  (lower is better)", fontsize=11)
    ax.set_title(
        "Feature Set Complexity vs. Model Performance",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    output_path = output_dir / "brier_vs_feature_count.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_performance_table(metadata_list: list[dict], output_dir: Path) -> None:
    """
    Saves a CSV summary table of model name, feature count, Brier score and
    best hyperparameters.

    Args:
        metadata_list: List of model metadata dicts.
        output_dir: Directory into which the CSV is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in metadata_list:
        params = m.get("best_parameters", {})
        rows.append({
            "model": m["model_name"],
            "model_type": m.get("model_type", ""),
            "n_features": m["n_features"],
            "brier_score": round(m["brier_score"], 6),
            "best_C": params.get("logreg__C", ""),
            "best_penalty": params.get("logreg__penalty", ""),
            "best_solver": params.get("logreg__solver", ""),
            "test_set_size": m.get("test_set_size", ""),
        })

    df = pd.DataFrame(rows)
    output_path = output_dir / "model_performance_table.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


# --- Entry point -----------------------------------------------------------

def run_model_performance_analysis(project_root: Path) -> None:
    """
    Orchestrates the full model performance analysis section.

    Args:
        project_root: Absolute path to the backend project root.
    """
    output_dir = project_root / "exploration" / "figures" / "Model_Performance"
    models_dir = project_root / "models"

    print("  Loading model metadata...")
    metadata_list = load_all_model_metadata(models_dir)

    if not metadata_list:
        print("  No model metadata found. Skipping model performance analysis.")
        return

    print("  Plotting Brier score comparison...")
    plot_brier_score_comparison(metadata_list, output_dir)

    print("  Plotting Brier score vs. feature count...")
    plot_brier_vs_feature_count(metadata_list, output_dir)

    print("  Saving performance table to CSV...")
    save_performance_table(metadata_list, output_dir)
