"""
Reliability diagrams and calibration metrics for trained xG models.

Reconstructs held-out test sets by replicating the exact train/test split
from the training pipeline, then evaluates Expected Calibration Error (ECE),
Maximum Calibration Error (MCE) and Average Calibration Error (ACE).

Individual two-panel reliability diagrams and a combined four-model
comparison figure are produced and saved to the figures directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# --- Configuration ---------------------------------------------------------

_N_BINS = 10
_TEST_SIZE = 0.2
_RANDOM_STATE = 104
_TARGET_COL = "target"

_MODEL_ORDER = ["basic_model", "situation_model", "shottype_model", "advanced_model"]

_COLOURS = {
    "basic_model": "#4C72B0",
    "situation_model": "#55A868",
    "shottype_model": "#DD8452",
    "advanced_model": "#C44E52",
    "reference": "#888888",
}


# --- Calibration metrics ---------------------------------------------------

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = _N_BINS,
) -> dict:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
    and Average Calibration Error (ACE) from predicted probabilities.

    ECE is the sample-weighted mean of per-bin absolute calibration error.
    MCE is the maximum per-bin absolute deviation.
    ACE is the unweighted mean per-bin absolute deviation.

    Args:
        y_true: Ground-truth binary labels as a NumPy array.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equal-width probability bins.

    Returns:
        Dict with keys ``ece``, ``mce`` and ``ace``.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    deviations: list[float] = []
    weights: list[int] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        deviations.append(abs(y_prob[mask].mean() - y_true[mask].mean()))
        weights.append(n)

    if not deviations:
        return {"ece": float("nan"), "mce": float("nan"), "ace": float("nan")}

    w = np.array(weights)
    d = np.array(deviations)

    return {
        "ece": float((w * d).sum() / w.sum()),
        "mce": float(d.max()),
        "ace": float(d.mean()),
    }


# --- Test-set reconstruction -----------------------------------------------

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
        df: Full preprocessed dataset.
        features: Feature columns used by the target model.
        target_col: Name of the binary target column.
        test_size: Proportion of data reserved for testing.
        random_state: Random seed — must match the training pipeline.

    Returns:
        Tuple of ``(X_test, y_test)``.
    """
    _, X_test, _, y_test = train_test_split(
        df[features],
        df[target_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return X_test, y_test


# --- Individual reliability diagram ----------------------------------------

def plot_individual_reliability_diagram(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
    n_bins: int = _N_BINS,
) -> dict:
    """
    Plots and saves a two-panel reliability diagram for a single model,
    annotated with ECE, MCE and ACE.

    The upper panel shows the calibration curve against the perfect-calibration
    diagonal; the lower panel shows the predicted-probability histogram.

    Args:
        model: Fitted sklearn-compatible estimator with ``predict_proba``.
        X_test: Feature matrix for the held-out test set.
        y_test: True binary labels for the held-out test set.
        model_name: Human-readable name used for the title and filename.
        output_dir: Directory in which the output figure is saved.
        n_bins: Number of bins for the calibration curve and histogram.

    Returns:
        Dict with ``model_name``, ``ece``, ``mce`` and ``ace``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(
        y_test, y_prob, n_bins=n_bins, strategy="uniform",
    )

    metrics = compute_calibration_metrics(np.array(y_test), y_prob, n_bins)

    colour = _COLOURS.get(model_name, "#1f77b4")

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.5, 1])
    ax_curve = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # --- Calibration curve panel ---
    ax_curve.plot(
        [0, 1], [0, 1],
        linestyle="--", lw=1.5,
        color=_COLOURS["reference"],
        label="Perfectly calibrated",
    )
    ax_curve.plot(
        mean_pred, frac_pos,
        marker="o", markersize=7, lw=2,
        color=colour,
        label=model_name,
    )

    # ±0.02 shaded uncertainty band around the calibration curve
    ax_curve.fill_between(
        mean_pred,
        np.clip(frac_pos - 0.02, 0, 1),
        np.clip(frac_pos + 0.02, 0, 1),
        alpha=0.15,
        color=colour,
        label="±0.02 band",
    )

    metrics_text = (
        f"ECE = {metrics['ece']:.4f}\n"
        f"MCE = {metrics['mce']:.4f}\n"
        f"ACE = {metrics['ace']:.4f}"
    )
    ax_curve.text(
        0.04, 0.96, metrics_text,
        transform=ax_curve.transAxes,
        fontsize=10, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#999", alpha=0.9),
    )

    ax_curve.set_xlim(0, 1)
    ax_curve.set_ylim(0, 1)
    ax_curve.set_xlabel("Mean Predicted xG", fontsize=11)
    ax_curve.set_ylabel("Observed Goal Rate", fontsize=11)
    ax_curve.set_title(
        f"Reliability Diagram — {model_name}",
        fontsize=13, fontweight="bold",
    )
    ax_curve.legend(loc="lower right", fontsize=10)
    ax_curve.grid(True, linestyle="--", alpha=0.5)

    # --- Prediction distribution histogram ---
    counts, _, bars = ax_hist.hist(
        y_prob, bins=n_bins, range=(0.0, 1.0),
        color=colour, edgecolor="white", linewidth=0.5, alpha=0.85,
    )
    ax_hist.bar_label(bars, labels=[f"{int(c):,}" for c in counts], fontsize=7.5, padding=3)
    ax_hist.set_ylim(0, max(counts) * 1.22)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Predicted xG", fontsize=11)
    ax_hist.set_ylabel("Shot Count", fontsize=11)
    ax_hist.grid(True, linestyle="--", alpha=0.5, axis="y")

    output_path = output_dir / f"reliability_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return {"model_name": model_name, **metrics}


# --- Combined four-model comparison ----------------------------------------

def plot_combined_calibration(
    calibration_data: list[dict],
    output_dir: Path,
) -> None:
    """
    Saves a single figure comparing calibration curves for all models on one
    set of axes, enabling direct visual comparison of calibration quality.

    Args:
        calibration_data: List of dicts, each with ``model_name``,
            ``mean_pred``, ``frac_pos`` and ``metrics`` keys.
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", lw=1.5,
        color=_COLOURS["reference"],
        label="Perfectly calibrated",
        zorder=2,
    )

    for entry in calibration_data:
        name = entry["model_name"]
        colour = _COLOURS.get(name, "#1f77b4")
        ece = entry["metrics"]["ece"]
        ax.plot(
            entry["mean_pred"],
            entry["frac_pos"],
            marker="o", markersize=6, lw=2,
            color=colour,
            label=f"{name}  (ECE = {ece:.4f})",
            zorder=3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted xG", fontsize=12)
    ax.set_ylabel("Observed Goal Rate", fontsize=12)
    ax.set_title(
        "Calibration Comparison — All Models",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = output_dir / "calibration_combined.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_calibration_metrics_table(all_metrics: list[dict], output_dir: Path) -> None:
    """
    Saves a CSV table of ECE, MCE and ACE for each model.

    Args:
        all_metrics: List of dicts with ``model_name``, ``ece``, ``mce``, ``ace``.
        output_dir: Directory into which the CSV is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    output_path = output_dir / "calibration_metrics.csv"
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"  Saved: {output_path}")


# --- Entry point -----------------------------------------------------------

def run_calibration_analysis(project_root: Path) -> None:
    """
    Orchestrates the full calibration analysis: loads models and the
    preprocessed dataset, reconstructs held-out test sets, plots individual
    reliability diagrams and a combined comparison figure.

    Args:
        project_root: Absolute path to the backend project root.
    """
    output_dir = project_root / "exploration" / "figures" / "Calibration"
    data_path = project_root / "data" / "preprocessed" / "preprocessed_shots.csv"
    models_root = project_root / "models"

    print(f"  Loading preprocessed data from {data_path} ...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"  Error: data not found at {data_path}. Skipping calibration analysis.")
        return

    print(f"  Data loaded — {len(df):,} rows.")

    all_metrics: list[dict] = []
    calibration_data: list[dict] = []

    for model_name in _MODEL_ORDER:
        model_path = models_root / model_name / "model.joblib"
        metadata_path = models_root / model_name / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            print(f"  Skipping '{model_name}' — missing artefacts.")
            continue

        print(f"  Processing: {model_name}")

        with open(metadata_path, "r") as fh:
            metadata = json.load(fh)

        features: list[str] = metadata.get("features", [])
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"  Skipping '{model_name}' — missing columns: {missing}")
            continue

        model = joblib.load(model_path)
        X_test, y_test = _reconstruct_test_set(df, features)

        metrics = plot_individual_reliability_diagram(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)

        # Collect data for the combined figure
        y_prob = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(
            y_test, y_prob, n_bins=_N_BINS, strategy="uniform",
        )
        calibration_data.append({
            "model_name": model_name,
            "mean_pred": mean_pred,
            "frac_pos": frac_pos,
            "metrics": {k: v for k, v in metrics.items() if k != "model_name"},
        })

    if calibration_data:
        print("  Plotting combined calibration figure...")
        plot_combined_calibration(calibration_data, output_dir)

    if all_metrics:
        print("  Saving calibration metrics table...")
        save_calibration_metrics_table(all_metrics, output_dir)

    print("  Calibration analysis complete.")
