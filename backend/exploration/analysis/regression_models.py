"""
xG predictive power analysis.

Tests whether accumulated past xG (or goals) over a rolling window
predicts future goals in a subsequent rolling window.  Pearson correlation
and Bonferroni-corrected significance thresholds are used throughout.

Produces per future window [2, 8, 16]:
  - Pearson correlation heatmap  (predictors x past-window sizes)
  - Predictor stability comparison chart across past-window sizes

Produces once (representative window: past=8, future=8):
  - Scatter panels with OLS regression lines for all predictors

Produces one combined chart:
  - xG (Advanced) vs. Goals head-to-head Pearson r across all future windows

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

# statsmodels is retained for OLS p-values; sklearn provides predictions.

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

# Past lookback window sizes — capped at 32 matches.
_PAST_WINDOWS = [1, 2, 4, 8, 16, 32]

# Future target windows: goals scored in the next N matches.
_FUTURE_WINDOWS = [2, 8, 16]

# Keep _WINDOWS as an alias used by heatmap/pivot helpers
_WINDOWS = _PAST_WINDOWS

# Representative windows for scatter panels (balanced past/future view).
_REPRESENTATIVE_WINDOW = 8
_REPRESENTATIVE_FUTURE_WINDOW = 8

_COLOURS = {
    "past_goals": "#888888",
    "past_xg_basic": "#4C72B0",
    "past_xg_situation": "#55A868",
    "past_xg_shottype": "#DD8452",
    "past_xg_advanced": "#C44E52",
}

# Expose a single _FUTURE_WINDOW alias so heatmap title helpers still work.
# This is set dynamically per iteration in run_xg_prediction_analysis but
# has a sensible default here for standalone use.
_FUTURE_WINDOW = _REPRESENTATIVE_FUTURE_WINDOW


# --- Core regression -------------------------------------------------------

def run_linear_regressions(lagged_df: pd.DataFrame) -> list[dict]:
    """
    Fits a separate OLS linear regression for each predictor against
    ``future_goals``, returning Pearson correlation alongside standard
    regression diagnostics.

    Args:
        lagged_df: DataFrame with rolling-window predictor columns and
            a ``future_goals`` target column.

    Returns:
        List of result dicts, one per predictor.  Dicts contain keys
        ``predictor``, ``Pearson Correlation``, ``Pearson P-value``,
        ``P-value (Coefficient)``, ``R-squared``, ``MAE``, ``RMSE``,
        ``Coefficient``, ``Intercept``, ``data`` and ``predictions``.
        On failure, a ``note`` key is included instead.
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
    future_window: int,
    metric: str = "Pearson Correlation",
) -> None:
    """
    Saves a Pearson correlation heatmap across all predictor/past-window
    combinations for a given future target window, with Bonferroni-corrected
    significance indicated by asterisks.

    Args:
        summary_df: DataFrame with columns ``predictor``, ``window_size``,
            ``Pearson Correlation`` and ``Pearson P-value``.
        output_dir: Directory into which the figure is saved.
        future_window: Number of future matches used as the target (for title).
        metric: Column name to use as the heatmap values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tests = len(summary_df)
    bonferroni_threshold = 0.05 / n_tests if n_tests > 0 else 0.05

    pivot = summary_df.pivot(index="predictor", columns="window_size", values=metric)
    pivot = pivot.reindex(index=_PREDICTORS, columns=_WINDOWS)
    pivot.index = [_PREDICTOR_LABELS.get(p, p) for p in pivot.index]

    # Significance pivot
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
        f"{metric} — Past Predictor (varying window) vs. Next {future_window} Matches' Goals\n"
        f"* = Bonferroni-significant (α/n = {bonferroni_threshold:.4f})",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Past Rolling Window Size (matches)", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    metric_slug = metric.lower().replace(" ", "_")
    output_path = output_dir / f"correlation_heatmap_future{future_window}_{metric_slug}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_regression_scatter_panels(
    results: list[dict],
    window_size: int,
    output_dir: Path,
    future_window: int = _REPRESENTATIVE_FUTURE_WINDOW,
) -> None:
    """
    Saves a grid of scatter plots with OLS regression lines, one per
    predictor, for a given past and future rolling window size.

    Args:
        results: Output of ``run_linear_regressions`` for this window.
        window_size: The past window size used (appears in the figure title).
        output_dir: Directory into which the figure is saved.
        future_window: The future target window used (appears in the title).
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
        f"Linear Regression: Predicting Goals in Next {future_window} Matches  "
        f"·  Past Window = {window_size} matches",
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
        ax.set_ylabel(f"Goals in Next {future_window} Matches", fontsize=9)
        ax.legend(fontsize=8)

        stats_text = (
            f"Pearson r = {result['Pearson Correlation']:.3f}\n"
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


def plot_predictor_comparison_panel(
    summary_df: pd.DataFrame,
    output_dir: Path,
    future_window: int = _REPRESENTATIVE_FUTURE_WINDOW,
) -> None:
    """
    Saves a single line chart showing how Pearson correlation for each
    predictor changes across past rolling window sizes for a fixed future
    target window.  The legend is placed in the lower-right corner to avoid
    overlap with rising curves.

    Args:
        summary_df: Aggregated results DataFrame with all window/predictor rows.
        output_dir: Directory into which the figure is saved.
        future_window: The future target window size (used in title and filename).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for predictor in _PREDICTORS:
        subset = summary_df[summary_df["predictor"] == predictor].sort_values("window_size")
        if subset.empty:
            continue
        ax.plot(
            subset["window_size"],
            subset["Pearson Correlation"],
            marker="o",
            markersize=5,
            lw=2,
            color=_COLOURS.get(predictor, "#888"),
            label=_PREDICTOR_LABELS.get(predictor, predictor),
        )

    ax.axhline(0, color="#888", lw=0.8, linestyle="--")
    ax.set_xlabel("Past Rolling Window Size (matches)", fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title("Pearson r", fontsize=12, fontweight="bold")
    # Legend placed in the lower-right so it does not overlap the upward-
    # trending correlation curves that cluster in the top portion.
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.FixedLocator(_PAST_WINDOWS))

    fig.suptitle(
        f"Predictor Stability Across Past Window Sizes  "
        f"·  Target = Goals in Next {future_window} Matches",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    output_path = output_dir / f"predictor_comparison_future{future_window}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_xg_vs_goals_headtohead(
    all_summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Saves a faceted line chart comparing Pearson r for ``past_xg_advanced``
    and ``past_goals`` across all past window sizes, with one subplot per
    future target window.

    The chart is the primary evidence that xG is a better predictor of
    future scoring than raw goals in most circumstances.

    Args:
        all_summary_df: Combined summary DataFrame with a ``future_window``
            column and rows for all future/past window combinations.
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    focus = ["past_goals", "past_xg_advanced"]
    future_windows = sorted(all_summary_df["future_window"].unique())
    n_panels = len(future_windows)

    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 6, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, fw in zip(axes, future_windows):
        subset_fw = all_summary_df[all_summary_df["future_window"] == fw]
        for predictor in focus:
            sub = subset_fw[subset_fw["predictor"] == predictor].sort_values("window_size")
            if sub.empty:
                continue
            ax.plot(
                sub["window_size"],
                sub["Pearson Correlation"],
                marker="o",
                markersize=6,
                lw=2.2,
                color=_COLOURS[predictor],
                label=_PREDICTOR_LABELS[predictor],
            )

        ax.axhline(0, color="#aaa", lw=0.8, linestyle="--")
        ax.set_title(f"Next {fw} Matches", fontsize=12, fontweight="bold")
        ax.set_xlabel("Past Window (matches)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Pearson r", fontsize=11)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
        ax.grid(linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(ticker.FixedLocator(_PAST_WINDOWS))

    fig.suptitle(
        "xG (Advanced) vs. Past Goals as Predictors of Future Scoring\n"
        "Higher Pearson r = stronger predictor  ·  Past window varies, "
        "future target window shown per panel",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path = output_dir / "xg_vs_goals_headtohead.png"
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
        "future_window", "window_size", "predictor", "predictor_label",
        "Pearson Correlation", "Pearson P-value",
        "R-squared", "MAE", "RMSE",
        "Coefficient", "Intercept",
        "bonferroni_significant",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values([c for c in ["future_window", "window_size", "predictor"] if c in df.columns])

    output_path = output_dir / "xg_prediction_statistical_summary.csv"
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"  Saved: {output_path}")


def save_correlation_pivot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Saves Pearson correlation pivots (predictors x windows) to CSV.

    Args:
        summary_df: Aggregated results DataFrame (may include a ``future_window`` column).
        output_dir: Directory into which the CSV is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # If multiple future windows are present, write one pivot per future window.
    if "future_window" in summary_df.columns:
        for fw in sorted(summary_df["future_window"].unique()):
            sub = summary_df[summary_df["future_window"] == fw]
            pivot = (
                sub.pivot(index="predictor", columns="window_size", values="Pearson Correlation")
                .reindex(index=_PREDICTORS, columns=_WINDOWS)
            )
            pivot.index = [_PREDICTOR_LABELS.get(p, p) for p in pivot.index]
            path = output_dir / f"correlation_pivot_pearson_future{fw}.csv"
            pivot.to_csv(path, float_format="%.4f")
            print(f"  Saved: {path}")
    else:
        pivot = (
            summary_df.pivot(index="predictor", columns="window_size", values="Pearson Correlation")
            .reindex(index=_PREDICTORS, columns=_WINDOWS)
        )
        pivot.index = [_PREDICTOR_LABELS.get(p, p) for p in pivot.index]
        path = output_dir / "correlation_pivot_pearson_correlation.csv"
        pivot.to_csv(path, float_format="%.4f")
        print(f"  Saved: {path}")


# --- Entry point -----------------------------------------------------------

def run_xg_prediction_analysis(
    project_root: Path,
    df_with_xg: pd.DataFrame,
) -> None:
    """
    Orchestrates the full xG predictive power analysis.

    Iterates over all combinations of past rolling windows (1–32) and
    future target windows (2, 8, 16).  For each future window a Pearson
    heatmap and predictor stability chart are produced.  Scatter panels are
    produced for the representative combination (past=8, future=8).  A
    combined head-to-head chart (xG Advanced vs. Goals) across all future
    windows is saved as the primary evidence plot.

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

    for future_window in _FUTURE_WINDOWS:
        fw_summary_rows: list[dict] = []

        for past_window in _PAST_WINDOWS:
            print(f"  Past window = {past_window}, future window = {future_window} ...")
            lagged_df = create_lagged_features(
                player_match_df,
                past_window_size=past_window,
                future_window_size=future_window,
            )

            if lagged_df.empty:
                print(
                    f"  No data after lagging for past={past_window}, "
                    f"future={future_window}. Skipping."
                )
                continue

            results = run_linear_regressions(lagged_df)

            for r in results:
                if "note" in r:
                    continue
                row = {
                    "future_window": future_window,
                    "window_size": past_window,
                    "predictor": r["predictor"],
                    "Pearson Correlation": r["Pearson Correlation"],
                    "Pearson P-value": r["Pearson P-value"],
                    "R-squared": r["R-squared"],
                    "MAE": r["MAE"],
                    "RMSE": r["RMSE"],
                    "Coefficient": r["Coefficient"],
                    "Intercept": r["Intercept"],
                }
                fw_summary_rows.append(row)
                all_summary_rows.append(row)

            # Scatter panels for the representative window combination only
            if (past_window == _REPRESENTATIVE_WINDOW
                    and future_window == _REPRESENTATIVE_FUTURE_WINDOW):
                print(
                    f"  Plotting scatter panels for "
                    f"past={past_window}, future={future_window}..."
                )
                plot_regression_scatter_panels(
                    results, past_window, output_dir, future_window=future_window,
                )

        if not fw_summary_rows:
            continue

        fw_df = pd.DataFrame(fw_summary_rows)

        print(f"  Plotting Pearson heatmap for future window = {future_window}...")
        plot_correlation_heatmap(
            fw_df, output_dir,
            future_window=future_window,
            metric="Pearson Correlation",
        )

        print(f"  Plotting predictor comparison for future window = {future_window}...")
        plot_predictor_comparison_panel(fw_df, output_dir, future_window=future_window)

    if not all_summary_rows:
        print("  No regression results to aggregate. Skipping remaining outputs.")
        return

    all_summary_df = pd.DataFrame(all_summary_rows)

    print("  Plotting xG vs Goals head-to-head comparison...")
    plot_xg_vs_goals_headtohead(all_summary_df, output_dir)

    print("  Saving statistical summary table...")
    save_statistical_summary(all_summary_rows, output_dir, len(all_summary_rows))

    print("  Saving correlation pivots to CSV...")
    save_correlation_pivot(all_summary_df, output_dir)

    print("  xG prediction analysis complete.")
