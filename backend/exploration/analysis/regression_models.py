"""
xG predictive power analysis.

Tests whether accumulated past xG (or goals) over a rolling window
predicts future goals in a subsequent rolling window.  Pearson and
Spearman correlations are computed for every predictor/window
combination, with Bonferroni-corrected significance thresholds.

Produces:
  - Correlation heatmap  (predictors x window sizes)
  - Scatter panels with regression lines (representative window)
  - Residual diagnostics for the best predictor
  - Statistical summary CSV
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analysis.data_aggregation import create_lagged_features, create_player_match_stats

# --- Constants -------------------------------------------------------------

_PREDICTORS = [
    "past_goals",
    "past_xg_basic",
    "past_xg_situation",
    "past_xg_shottype",
    "past_xg_advanced",
]

_PREDICTOR_LABELS = {
    "past_goals": "Past Goals",
    "past_xg_basic": "Past xG (Basic)",
    "past_xg_situation": "Past xG (Situation)",
    "past_xg_shottype": "Past xG (Shot Type)",
    "past_xg_advanced": "Past xG (Advanced)",
}

_WINDOWS = [1, 2, 4, 8, 16, 32, 64]

# Representative window for scatter and residual panels
_REPRESENTATIVE_WINDOW = 8

_COLOURS = {
    "past_goals": "#888888",
    "past_xg_basic": "#4C72B0",
    "past_xg_situation": "#55A868",
    "past_xg_shottype": "#DD8452",
    "past_xg_advanced": "#C44E52",
}


# --- Core regression -------------------------------------------------------

def run_linear_regressions(lagged_df: pd.DataFrame) -> list[dict]:
    """
    Fits a separate OLS linear regression for each predictor against
    ``future_goals``, returning Pearson and Spearman correlations alongside
    standard regression diagnostics.

    Args:
        lagged_df: DataFrame with rolling-window predictor columns and
            a ``future_goals`` target column.

    Returns:
        List of result dicts, one per predictor.  Dicts contain keys
        ``predictor``, ``Pearson Correlation``, ``Spearman Correlation``,
        ``P-value (Coefficient)``, ``Spearman P-value``, ``R-squared``,
        ``MAE``, ``RMSE``, ``Coefficient``, ``Intercept``, ``data`` and
        ``predictions``.  On failure, a ``note`` key is included instead.
    """
    results = []

    for predictor in _PREDICTORS:
        if predictor not in lagged_df.columns or "future_goals" not in lagged_df.columns:
            results.append({"predictor": predictor, "note": "Column missing"})
            continue

        tmp = lagged_df[[predictor, "future_goals"]].dropna()
        if len(tmp) < 10:
            results.append({"predictor": predictor, "note": "Insufficient data"})
            continue

        X = tmp[[predictor]]
        y = tmp["future_goals"]

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(X.squeeze(), y)

        # Spearman rank correlation
        spearman_r, spearman_p = stats.spearmanr(X.squeeze(), y)

        # OLS via statsmodels for p-value on the coefficient
        X_sm = sm.add_constant(X)
        sm_res = sm.OLS(y, X_sm).fit()
        coeff_p = sm_res.pvalues[predictor]

        # sklearn for predictions and error metrics
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        results.append({
            "predictor": predictor,
            "Pearson Correlation": pearson_r,
            "Pearson P-value": pearson_p,
            "Spearman Correlation": spearman_r,
            "Spearman P-value": spearman_p,
            "P-value (Coefficient)": coeff_p,
            "R-squared": r2_score(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "Coefficient": model.coef_[0],
            "Intercept": model.intercept_,
            "data": tmp,
            "predictions": y_pred,
        })

    return results


# --- Visualisations --------------------------------------------------------

def plot_correlation_heatmap(
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric: str = "Pearson Correlation",
) -> None:
    """
    Saves a heatmap of the chosen correlation metric across all
    predictor/window combinations, with Bonferroni-corrected significance
    indicated by asterisks.

    Args:
        summary_df: DataFrame with columns ``predictor``, ``window_size``,
            ``Pearson Correlation``, ``Pearson P-value``,
            ``Spearman Correlation``, ``Spearman P-value``.
        output_dir: Directory into which the figure is saved.
        metric: Column name to use as the heatmap values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tests = len(summary_df)
    bonferroni_threshold = 0.05 / n_tests if n_tests > 0 else 0.05

    pivot = summary_df.pivot(index="predictor", columns="window_size", values=metric)
    pivot = pivot.reindex(index=_PREDICTORS, columns=_WINDOWS)
    pivot.index = [_PREDICTOR_LABELS.get(p, p) for p in pivot.index]

    # Significance mask (True where p > Bonferroni threshold)
    p_col = metric.replace("Correlation", "P-value")
    pivot_p = summary_df.pivot(index="predictor", columns="window_size", values=p_col)
    pivot_p = pivot_p.reindex(index=_PREDICTORS, columns=_WINDOWS)
    pivot_p.index = [_PREDICTOR_LABELS.get(p, p) for p in pivot_p.index]

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-0.1,
        vmax=0.5,
        annot=False,
        linewidths=0.5,
        linecolor="#dddddd",
        cbar_kws={"label": metric, "shrink": 0.8},
    )

    for row_i, row_label in enumerate(pivot.index):
        for col_i, col_label in enumerate(pivot.columns):
            val = pivot.iloc[row_i, col_i]
            p_val = pivot_p.iloc[row_i, col_i]

            if pd.isna(val):
                continue

            significance = ""
            if not pd.isna(p_val):
                if p_val < bonferroni_threshold:
                    significance = "*"
                if p_val < bonferroni_threshold / 10:
                    significance = "**"

            cell_colour = "white" if abs(val) > 0.35 else "black"
            ax.text(
                col_i + 0.5,
                row_i + 0.42,
                f"{val:.3f}",
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=cell_colour,
            )
            if significance:
                ax.text(
                    col_i + 0.5,
                    row_i + 0.70,
                    significance,
                    ha="center", va="center",
                    fontsize=8, color=cell_colour,
                )

    ax.set_title(
        f"{metric} — Past Predictor vs. Future Goals\n"
        f"* = Bonferroni-significant (α/n = {bonferroni_threshold:.4f})",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Rolling Window Size (matches)", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    metric_slug = metric.lower().replace(" ", "_")
    output_path = output_dir / f"correlation_heatmap_{metric_slug}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_regression_scatter_panels(
    results: list[dict],
    window_size: int,
    output_dir: Path,
) -> None:
    """
    Saves a grid of scatter plots with OLS regression lines, one per
    predictor, for a given rolling window size.

    Args:
        results: Output of ``run_linear_regressions`` for this window.
        window_size: The window size used (appears in the figure title).
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = [r for r in results if "note" not in r]
    if not valid:
        print(f"  No valid regression results for window {window_size}.")
        return

    # Sort by predictor order
    order = {p: i for i, p in enumerate(_PREDICTORS)}
    valid.sort(key=lambda r: order.get(r["predictor"], 99))

    n_cols = min(len(valid), 3)
    n_rows = (len(valid) + n_cols - 1) // n_cols

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle(
        f"Linear Regression: Predicting Future Goals  ·  Window = {window_size} matches",
        fontsize=15, fontweight="bold",
    )

    # Shared axis limits across all panels for comparability
    all_x = np.concatenate([r["data"][r["predictor"]].values for r in valid])
    all_y = np.concatenate([r["data"]["future_goals"].values for r in valid])
    x_pad = (all_x.max() - all_x.min()) * 0.05
    y_pad = (all_y.max() - all_y.min()) * 0.05
    x_lim = (all_x.min() - x_pad, all_x.max() + x_pad)
    y_lim = (all_y.min() - y_pad, all_y.max() + y_pad)

    for i, result in enumerate(valid):
        ax = axes_flat[i]
        predictor = result["predictor"]
        colour = _COLOURS.get(predictor, "dodgerblue")

        ax.scatter(
            result["data"][predictor],
            result["data"]["future_goals"],
            alpha=0.35, s=18,
            color=colour, edgecolors="none",
            label="Observations",
        )
        ax.plot(
            result["data"][predictor],
            result["predictions"],
            color="#222",
            linewidth=2,
            label="OLS fit",
        )

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(
            _PREDICTOR_LABELS.get(predictor, predictor),
            fontsize=11, fontweight="medium",
        )
        ax.set_xlabel(predictor, fontsize=9)
        ax.set_ylabel("Future Goals", fontsize=9)
        ax.legend(fontsize=8)

        stats_text = (
            f"Pearson r = {result['Pearson Correlation']:.3f}\n"
            f"Spearman ρ = {result['Spearman Correlation']:.3f}\n"
            f"R² = {result['R-squared']:.4f}"
        )
        ax.text(
            0.97, 0.05, stats_text,
            transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#aaa", alpha=0.9),
        )

    for j in range(len(valid), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / f"regression_scatter_window_{window_size}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_residual_diagnostics(
    result: dict,
    window_size: int,
    output_dir: Path,
) -> None:
    """
    Saves a four-panel residual diagnostic figure for a single predictor,
    comprising a residuals-vs-fitted plot, Q-Q plot, residual histogram and
    scale-location plot.

    Args:
        result: Single predictor result dict from ``run_linear_regressions``.
        window_size: The window size used (appears in the figure title).
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = result["predictor"]
    y_true = result["data"]["future_goals"].values
    y_pred = result["predictions"]
    residuals = y_true - y_pred
    std_resid = residuals / (residuals.std() + 1e-9)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colour = _COLOURS.get(predictor, "#4C72B0")

    # 1. Residuals vs. Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.25, s=15, color=colour, edgecolors="none")
    ax.axhline(0, color="#d62728", lw=1.5, linestyle="--")
    ax.set_xlabel("Fitted Values", fontsize=10)
    ax.set_ylabel("Residuals", fontsize=10)
    ax.set_title("Residuals vs. Fitted", fontsize=11, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)

    # 2. Q-Q plot
    ax = axes[0, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.3, s=15, color=colour, edgecolors="none")
    qq_line_x = np.array([min(osm), max(osm)])
    ax.plot(qq_line_x, slope * qq_line_x + intercept, color="#d62728", lw=1.5)
    ax.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax.set_ylabel("Sample Quantiles", fontsize=10)
    ax.set_title("Q-Q Plot  (Normality Check)", fontsize=11, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)

    # 3. Histogram of residuals
    ax = axes[1, 0]
    ax.hist(residuals, bins=40, color=colour, edgecolor="white", alpha=0.8)
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    pdf = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
    ax2 = ax.twinx()
    ax2.plot(x_range, pdf, color="#d62728", lw=1.8, label="Normal PDF")
    ax2.set_ylabel("Density", fontsize=9, color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")
    ax.set_xlabel("Residual", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Residual Distribution", fontsize=11, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4, axis="y")

    # 4. Scale-location plot (√|std residuals| vs. fitted)
    ax = axes[1, 1]
    sqrt_abs_resid = np.sqrt(np.abs(std_resid))
    ax.scatter(y_pred, sqrt_abs_resid, alpha=0.25, s=15, color=colour, edgecolors="none")
    # Lowess smoothing line
    try:
        smoothed = sm.nonparametric.lowess(sqrt_abs_resid, y_pred, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="#d62728", lw=1.8, label="LOWESS")
        ax.legend(fontsize=9)
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlabel("Fitted Values", fontsize=10)
    ax.set_ylabel("√|Standardised Residuals|", fontsize=10)
    ax.set_title("Scale-Location", fontsize=11, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Residual Diagnostics — {_PREDICTOR_LABELS.get(predictor, predictor)}"
        f"  ·  Window = {window_size} matches",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / f"residual_diagnostics_{predictor}_window_{window_size}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_predictor_comparison_panel(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Saves a line chart showing how Pearson and Spearman correlations for
    each predictor change across rolling window sizes, making it easy to
    assess whether xG metrics stabilise faster than raw goals.

    Args:
        summary_df: Aggregated results DataFrame with all window/predictor rows.
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, (metric, label) in zip(
        axes,
        [
            ("Pearson Correlation", "Pearson r"),
            ("Spearman Correlation", "Spearman ρ"),
        ],
    ):
        for predictor in _PREDICTORS:
            subset = summary_df[summary_df["predictor"] == predictor].sort_values("window_size")
            if subset.empty:
                continue
            ax.plot(
                subset["window_size"],
                subset[metric],
                marker="o",
                markersize=5,
                lw=2,
                color=_COLOURS.get(predictor, "#888"),
                label=_PREDICTOR_LABELS.get(predictor, predictor),
            )

        ax.axhline(0, color="#888", lw=0.8, linestyle="--")
        ax.set_xlabel("Rolling Window Size (matches)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(
            ticker.FixedLocator(_WINDOWS)
        )

    fig.suptitle(
        "Predictor Stability Across Rolling Window Sizes",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    output_path = output_dir / "predictor_comparison_by_window.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# --- Statistical summary ---------------------------------------------------

def save_statistical_summary(
    summary_rows: list[dict],
    output_dir: Path,
    n_total_tests: int,
) -> None:
    """
    Saves a CSV table with correlation coefficients, p-values and
    Bonferroni-corrected significance flags for every predictor/window pair.

    Args:
        summary_rows: List of dicts, one per (predictor, window) combination.
        output_dir: Directory into which the CSV is saved.
        n_total_tests: Total number of tests (used for Bonferroni correction).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(summary_rows)
    bonferroni = 0.05 / n_total_tests if n_total_tests > 0 else 0.05
    df["bonferroni_significant"] = df["Pearson P-value"] < bonferroni
    df["predictor_label"] = df["predictor"].map(_PREDICTOR_LABELS)

    col_order = [
        "window_size", "predictor", "predictor_label",
        "Pearson Correlation", "Pearson P-value",
        "Spearman Correlation", "Spearman P-value",
        "R-squared", "MAE", "RMSE",
        "Coefficient", "Intercept",
        "bonferroni_significant",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["window_size", "predictor"])

    output_path = output_dir / "xg_prediction_statistical_summary.csv"
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"  Saved: {output_path}")


def save_correlation_pivot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Saves Pearson correlation pivots (predictors x windows) to CSV.

    Args:
        summary_df: Aggregated results DataFrame.
        output_dir: Directory into which the CSV is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("Pearson Correlation", "Spearman Correlation"):
        pivot = (
            summary_df.pivot(index="predictor", columns="window_size", values=metric)
            .reindex(index=_PREDICTORS, columns=_WINDOWS)
        )
        pivot.index = [_PREDICTOR_LABELS.get(p, p) for p in pivot.index]
        slug = metric.lower().replace(" ", "_")
        path = output_dir / f"correlation_pivot_{slug}.csv"
        pivot.to_csv(path, float_format="%.4f")
        print(f"  Saved: {path}")


# --- Entry point -----------------------------------------------------------

def run_xg_prediction_analysis(
    project_root: Path,
    df_with_xg: pd.DataFrame,
) -> None:
    """
    Orchestrates the full xG predictive power analysis.

    Runs rolling-window regressions for every predictor/window combination,
    produces all visualisations and saves summary tables.

    Args:
        project_root: Absolute path to the backend project root.
        df_with_xg: Shot-level DataFrame with model xG predictions appended,
            as returned by the preprocessing pipeline.
    """
    output_dir = project_root / "exploration" / "figures" / "xG_Prediction"

    print("  Aggregating player match statistics...")
    player_match_df = create_player_match_stats(df_with_xg)
    print(f"  Player match stats: {len(player_match_df):,} rows.")

    all_summary_rows: list[dict] = []

    for window in _WINDOWS:
        print(f"  Window = {window} ...")
        lagged_df = create_lagged_features(player_match_df, window, window)

        if lagged_df.empty:
            print(f"  No data after lagging for window {window}. Skipping.")
            continue

        results = run_linear_regressions(lagged_df)

        for r in results:
            if "note" in r:
                continue
            all_summary_rows.append({
                "window_size": window,
                "predictor": r["predictor"],
                "Pearson Correlation": r["Pearson Correlation"],
                "Pearson P-value": r["Pearson P-value"],
                "Spearman Correlation": r["Spearman Correlation"],
                "Spearman P-value": r["Spearman P-value"],
                "R-squared": r["R-squared"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "Coefficient": r["Coefficient"],
                "Intercept": r["Intercept"],
            })

        # Scatter panels for the representative window
        if window == _REPRESENTATIVE_WINDOW:
            print(f"  Plotting scatter panels for window {window}...")
            plot_regression_scatter_panels(results, window, output_dir)

            # Residual diagnostics for the best predictor
            best = next(
                (r for r in results
                 if r.get("predictor") == "past_xg_advanced" and "note" not in r),
                None,
            )
            if best:
                print("  Plotting residual diagnostics for past_xg_advanced...")
                plot_residual_diagnostics(best, window, output_dir)

    if not all_summary_rows:
        print("  No regression results to aggregate. Skipping remaining outputs.")
        return

    summary_df = pd.DataFrame(all_summary_rows)

    print("  Plotting Pearson correlation heatmap...")
    plot_correlation_heatmap(summary_df, output_dir, metric="Pearson Correlation")

    print("  Plotting Spearman correlation heatmap...")
    plot_correlation_heatmap(summary_df, output_dir, metric="Spearman Correlation")

    print("  Plotting predictor comparison panel...")
    plot_predictor_comparison_panel(summary_df, output_dir)

    print("  Saving statistical summary table...")
    save_statistical_summary(all_summary_rows, output_dir, len(all_summary_rows))

    print("  Saving correlation pivots to CSV...")
    save_correlation_pivot(summary_df, output_dir)

    print("  xG prediction analysis complete.")
