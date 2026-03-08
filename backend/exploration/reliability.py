"""
Generates reliability diagrams (calibration curves) for trained xG models.

Each model's held-out test set is reconstructed by replicating the exact
train/test split from the training pipeline, ensuring evaluation is performed
on genuinely unseen data.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


# --- Configuration -----------------------------------------------------------

_N_BINS = 10
_TEST_SIZE = 0.2
_RANDOM_STATE = 104
_TARGET_COL = "target"

_COLOURS = {
    "curve": "#1f77b4",
    "reference": "#d62728",
    "histogram": "#aec7e8",
}


# --- Core plotting function --------------------------------------------------

def plot_reliability_diagram(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
    n_bins: int = _N_BINS,
) -> None:
    """
    Plots and saves a two-panel reliability diagram for a single model.

    The top panel shows the calibration curve (mean predicted probability vs.
    fraction of positives) against a perfectly calibrated reference line.
    The bottom panel shows a histogram of predicted probabilities, indicating
    how often predictions fall within each confidence bin.

    Args:
        model: A fitted sklearn-compatible estimator with a ``predict_proba`` method.
        X_test (pd.DataFrame): Feature matrix for the held-out test set.
        y_test (pd.Series): True binary labels for the held-out test set.
        model_name (str): Human-readable name used for the plot title and filename.
        output_dir (Path): Directory in which the output figure will be saved.
        n_bins (int): Number of bins for both the calibration curve and histogram.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_probs = model.predict_proba(X_test)[:, 1]

    fraction_of_positives, mean_predicted_prob = calibration_curve(
        y_test,
        predicted_probs,
        n_bins=n_bins,
        strategy="uniform",
    )

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.5, 1])

    # --- Top panel: calibration curve ---
    ax_curve = fig.add_subplot(gs[0])

    ax_curve.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color=_COLOURS["reference"],
        label="Perfectly calibrated",
    )
    ax_curve.plot(
        mean_predicted_prob,
        fraction_of_positives,
        marker="o",
        markersize=6,
        linewidth=2,
        color=_COLOURS["curve"],
        label=model_name,
    )

    ax_curve.set_xlim(0.0, 1.0)
    ax_curve.set_ylim(0.0, 1.0)
    ax_curve.set_xlabel("Mean Predicted xG", fontsize=11)
    ax_curve.set_ylabel("Observed Goal Rate", fontsize=11)
    ax_curve.set_title(f"Reliability Diagram — {model_name}", fontsize=13, fontweight="bold")
    ax_curve.legend(loc="upper left", fontsize=10)
    ax_curve.grid(True, linestyle="--", alpha=0.5)

    # --- Bottom panel: prediction distribution histogram ---
    ax_hist = fig.add_subplot(gs[1])

    counts, _, bars = ax_hist.hist(
        predicted_probs,
        bins=n_bins,
        range=(0.0, 1.0),
        color=_COLOURS["histogram"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax_hist.bar_label(
        bars,
        labels=[f"{int(c):,}" for c in counts],
        fontsize=8,
        padding=3,
    )

    # Extend y-axis ceiling to prevent count labels from clipping against the top.
    ax_hist.set_ylim(0, max(counts) * 1.18)
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("Predicted xG", fontsize=11)
    ax_hist.set_ylabel("Shot Count", fontsize=11)
    ax_hist.grid(True, linestyle="--", alpha=0.5, axis="y")

    output_path = output_dir / f"reliability_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}")


# --- Helpers -----------------------------------------------------------------

def _reconstruct_test_set(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = _TARGET_COL,
    test_size: float = _TEST_SIZE,
    random_state: int = _RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reconstructs the held-out test split by replicating the exact parameters
    used during model training.

    Args:
        df (pd.DataFrame): The full preprocessed dataset.
        features (list[str]): Feature columns used by the target model.
        target_col (str): Name of the binary target column.
        test_size (float): Proportion of data reserved for testing.
        random_state (int): Random seed — must match the training pipeline.

    Returns:
        tuple[pd.DataFrame, pd.Series]: ``(X_test, y_test)`` as a feature
        matrix and label vector respectively.
    """
    X = df[features]
    y = df[target_col]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_test, y_test


# --- Entry point -------------------------------------------------------------

def main() -> None:
    """
    Loads each trained model and its metadata, reconstructs the held-out test
    set, and saves a reliability diagram to the figures directory.
    """
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "preprocessed" / "preprocessed_shots.csv"
    models_root = project_root / "models"
    figures_dir = Path(__file__).resolve().parent / "figures"

    print(f"Loading data from {data_path} ...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: preprocessed data not found at {data_path}")
        return

    print(f"Data loaded — {len(df):,} rows.\n")

    model_dirs = sorted(p for p in models_root.iterdir() if p.is_dir())

    if not model_dirs:
        print(f"No model directories found under {models_root}.")
        return

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_path = model_dir / "model.joblib"
        metadata_path = model_dir / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            print(f"Skipping '{model_name}' — missing model.joblib or metadata.json.")
            continue

        print(f"Processing: {model_name}")

        with open(metadata_path, "r") as fh:
            metadata = json.load(fh)

        features: list[str] = metadata.get("features", [])
        missing = [f for f in features if f not in df.columns]

        if missing:
            print(f"  Skipping — missing feature columns: {missing}")
            continue

        model = joblib.load(model_path)
        X_test, y_test = _reconstruct_test_set(df, features)

        plot_reliability_diagram(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            output_dir=figures_dir,
        )

    print("\nDone. All reliability diagrams saved.")


if __name__ == "__main__":
    main()
