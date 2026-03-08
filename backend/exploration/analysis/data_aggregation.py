"""
Shot volume aggregation and player-level match statistics.

Produces league-by-season heatmap tables from raw shot data, and aggregates
shot-level records into per-player, per-match summaries with rolling-window
lagged features for xG predictive modelling experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- League ordering and display labels ------------------------------------

_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]

_LEAGUE_LABELS = {
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
}

_SEASON_RANGE = list(range(2014, 2026))


# --- Shot volume analysis --------------------------------------------------

def build_shot_volume_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a league x season pivot table of total shot counts.

    Args:
        df: Raw shot DataFrame with ``league`` and ``season`` columns.

    Returns:
        DataFrame indexed by league display name, columns = integer seasons.
    """
    pivot = (
        df.groupby(["league", "season"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=_LEAGUES, columns=_SEASON_RANGE, fill_value=0)
    )
    pivot.index = [_LEAGUE_LABELS.get(lg, lg) for lg in pivot.index]
    return pivot


def build_goal_volume_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a league x season pivot table of total goal counts.

    Args:
        df: Raw shot DataFrame with ``league``, ``season`` and ``result`` columns.

    Returns:
        DataFrame indexed by league display name, columns = integer seasons.
    """
    goals = df[df["result"] == "Goal"]
    pivot = (
        goals.groupby(["league", "season"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=_LEAGUES, columns=_SEASON_RANGE, fill_value=0)
    )
    pivot.index = [_LEAGUE_LABELS.get(lg, lg) for lg in pivot.index]
    return pivot


def _add_totals(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Appends a 'Total' column (row sums) and a 'Total' row (column sums)
    to a league x season pivot table.

    Args:
        pivot: The source pivot DataFrame.

    Returns:
        A new DataFrame with the extra totals row and column appended.
    """
    out = pivot.copy()
    out["Total"] = out.sum(axis=1)
    totals_row = out.sum(axis=0)
    totals_row.name = "Total"
    return pd.concat([out, totals_row.to_frame().T])


def plot_shot_volume_heatmap(
    pivot_shots: pd.DataFrame,
    pivot_goals: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Renders a colour-coded heatmap of shot volumes (leagues x seasons) with
    goal counts annotated as secondary text within each cell.  A 'Total'
    column and row are appended to show aggregate volumes.

    Args:
        pivot_shots: League x season pivot of shot counts.
        pivot_goals: League x season pivot of goal counts.
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add totals before rendering
    shots_with_totals = _add_totals(pivot_shots)
    goals_with_totals = _add_totals(pivot_goals)

    n_rows, n_cols = shots_with_totals.shape
    fig, ax = plt.subplots(figsize=(n_cols * 1.25, max(5, n_rows * 0.95)))

    # Colour only the inner cells (excluding the totals row/column).
    # The totals are shown in a neutral grey so they don't distort the colour scale.
    colour_data = shots_with_totals.copy().astype(float)
    colour_data.iloc[-1, :] = np.nan   # Totals row — uncoloured
    colour_data.iloc[:, -1] = np.nan   # Totals column — uncoloured

    sns.heatmap(
        colour_data,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="#dddddd",
        annot=False,
        cbar_kws={"label": "Shots per Season", "shrink": 0.8},
        mask=colour_data.isna(),
    )

    # Fill totals cells with a neutral grey background manually
    for col_i in range(n_cols):
        ax.add_patch(
            plt.Rectangle(
                (col_i, n_rows - 1), 1, 1,
                color="#cccccc", zorder=1,
            )
        )
    for row_i in range(n_rows):
        ax.add_patch(
            plt.Rectangle(
                (n_cols - 1, row_i), 1, 1,
                color="#cccccc", zorder=1,
            )
        )
    # Corner cell (Total/Total) slightly darker
    ax.add_patch(
        plt.Rectangle((n_cols - 1, n_rows - 1), 1, 1, color="#aaaaaa", zorder=1)
    )

    max_shots = int(pivot_shots.values.max())

    for row_i in range(n_rows):
        for col_i in range(n_cols):
            shots = int(shots_with_totals.iloc[row_i, col_i])
            goals = int(goals_with_totals.iloc[row_i, col_i])

            is_total = (row_i == n_rows - 1) or (col_i == n_cols - 1)
            text_colour = "#222" if is_total else (
                "white" if shots > max_shots * 0.65 else "black"
            )
            fw = "bold"

            ax.text(
                col_i + 0.5,
                row_i + 0.38,
                f"{shots:,}",
                ha="center", va="center",
                fontsize=7, fontweight=fw,
                color=text_colour, zorder=5,
            )
            ax.text(
                col_i + 0.5,
                row_i + 0.70,
                f"({goals:,}G)",
                ha="center", va="center",
                fontsize=5.5, color=text_colour, zorder=5,
            )

    ax.set_title(
        "Shot & Goal Volume by League and Season  ·  2014–2025\n"
        "Cell format: shots   (goals in parentheses)   ·   'Total' column/row shows aggregates",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

    plt.tight_layout()
    output_path = output_dir / "shot_volume_heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_shot_volume_tables(
    pivot_shots: pd.DataFrame,
    pivot_goals: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Saves shot and goal volume pivot tables as separate CSVs.

    Args:
        pivot_shots: League x season shot count pivot.
        pivot_goals: League x season goal count pivot.
        output_dir: Directory into which the CSVs are saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shots_path = output_dir / "shot_volume_by_league_season.csv"
    goals_path = output_dir / "goal_volume_by_league_season.csv"

    pivot_shots.to_csv(shots_path)
    pivot_goals.to_csv(goals_path)

    print(f"  Saved: {shots_path}")
    print(f"  Saved: {goals_path}")


def run_shot_volume_analysis(project_root: Path, raw_df: pd.DataFrame) -> None:
    """
    Orchestrates the shot volume analysis section.

    Args:
        project_root: Absolute path to the backend project root.
        raw_df: Pre-loaded raw shot DataFrame containing all original columns.
    """
    output_dir = project_root / "exploration" / "figures" / "Shot_Volume"

    print("  Building shot/goal volume pivots...")
    pivot_shots = build_shot_volume_pivot(raw_df)
    pivot_goals = build_goal_volume_pivot(raw_df)

    print("  Plotting shot volume heatmap...")
    plot_shot_volume_heatmap(pivot_shots, pivot_goals, output_dir)

    print("  Saving volume tables to CSV...")
    save_shot_volume_tables(pivot_shots, pivot_goals, output_dir)


# --- Player-level match statistics (consumed by xG prediction analysis) ----

def create_player_match_stats(df_with_xg: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates shot-level data into per-player, per-match summary statistics.

    Args:
        df_with_xg: Shot-level DataFrame with model xG predictions appended.

    Returns:
        DataFrame with one row per (match_id, player_id) combination.
    """
    player_match_df = df_with_xg.groupby(
        ["match_id", "player_id", "h_a", "date"]
    ).agg(
        total_xg_basic_match=("xG_basic_model", "sum"),
        total_xg_situation_match=("xG_situation_model", "sum"),
        total_xg_shottype_match=("xG_shottype_model", "sum"),
        total_xg_advanced_match=("xG_advanced_model", "sum"),
        total_goals_match=("result", lambda x: (x == "Goal").sum()),
    ).reset_index()

    return player_match_df.sort_values(
        by=["player_id", "date"]
    ).reset_index(drop=True)


def create_lagged_features(
    player_match_df: pd.DataFrame,
    past_window_size: int = 5,
    future_window_size: int = 5,
) -> pd.DataFrame:
    """
    Derives rolling past xG/goal totals and a forward-looking future-goals
    target for each player, supporting predictive regression experiments.

    Args:
        player_match_df: Per-player, per-match aggregated DataFrame.
        past_window_size: Matches to look back over for predictor sums.
        future_window_size: Matches to look ahead for the target sum.

    Returns:
        DataFrame with ``past_*`` predictor columns and ``future_goals``
        target column; rows where either window is incomplete are dropped.
    """
    df = player_match_df.sort_values(by=["player_id", "date"]).copy()

    _xg_col_map = {
        "total_xg_basic_match": "past_xg_basic",
        "total_xg_situation_match": "past_xg_situation",
        "total_xg_shottype_match": "past_xg_shottype",
        "total_xg_advanced_match": "past_xg_advanced",
        "total_goals_match": "past_goals",
    }

    for src_col, dst_col in _xg_col_map.items():
        df[dst_col] = (
            df.groupby("player_id")[src_col]
            .rolling(window=past_window_size, closed="left")
            .sum()
            .reset_index(level=0, drop=True)
        )

    # Future target: sum of goals over the next future_window_size matches.
    df["future_goals"] = (
        df.groupby("player_id")["total_goals_match"]
        .rolling(window=future_window_size, closed="right")
        .sum()
        .shift(periods=-future_window_size)
        .reset_index(level=0, drop=True)
    )

    return df.dropna().reset_index(drop=True)