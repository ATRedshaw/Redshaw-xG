"""
Exploration orchestrator.

Loads raw and preprocessed data once, then runs all analysis modules in
sequence, saving figures and tables under exploration/figures/<Category>/.

Designed to be run directly or invoked from run_pipeline.sh after the
main modelling pipeline completes.

Usage (from backend/):
    python exploration/explore.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# Ensure exploration/ is on sys.path so sub-modules import correctly
# regardless of the working directory used by the caller.
_EXPLORATION_DIR = Path(__file__).resolve().parent
if str(_EXPLORATION_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORATION_DIR))

from analysis.calibration import run_calibration_analysis
from analysis.data_aggregation import run_shot_volume_analysis
from analysis.feature_engineering import run_feature_engineering_analysis
from analysis.model_performance import run_model_performance_analysis
from analysis.regression_models import run_xg_prediction_analysis
from preprocessing.preprocessing import (
    apply_models_and_get_xg,
    load_raw_data_with_metadata,
    preprocess_for_prediction,
)


def _run_section(title: str, fn, *args, **kwargs) -> bool:
    """
    Calls ``fn(*args, **kwargs)`` with timing and graceful error handling.

    Args:
        title: Human-readable section name printed to stdout.
        fn: Callable to invoke.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        True if the section completed without exception, False otherwise.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    t0 = time.perf_counter()
    try:
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  Completed in {elapsed:.1f}s")
        return True
    except Exception:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        print(f"  FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return False


def main() -> None:
    """
    Runs the full exploration pipeline, producing all figures and tables.

    Data is loaded once at the top level and passed to each analysis module
    to avoid redundant I/O.
    """
    overall_start = time.perf_counter()

    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    models_dir = project_root / "models"

    print("\n" + "=" * 60)
    print("  xG Model Exploration Pipeline")
    print(f"  Project root: {project_root}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load raw data (used for shot volume and xG prediction analysis).
    # ------------------------------------------------------------------
    print("\nLoading raw shot data...")
    raw_df = load_raw_data_with_metadata(raw_data_dir)
    if raw_df.empty:
        print("Error: no raw data loaded. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Engineer features and apply models (for xG prediction analysis).
    #    This is the most expensive step; done once and shared.
    # ------------------------------------------------------------------
    df_with_xg = None
    try:
        print("\nEngineering features and applying models...")
        features_df = preprocess_for_prediction(raw_df)
        df_with_xg = apply_models_and_get_xg(raw_df, features_df, models_dir)
        print(f"  xG predictions appended — {len(df_with_xg):,} shots.")
    except Exception:  # noqa: BLE001
        print("  Warning: could not generate xG predictions. "
              "xG prediction analysis will be skipped.")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 3. Run each analysis section.
    # ------------------------------------------------------------------
    results: dict[str, bool] = {}

    results["Shot Volume"] = _run_section(
        "Shot Volume Analysis",
        run_shot_volume_analysis,
        project_root,
        raw_df,
    )

    results["Feature Engineering"] = _run_section(
        "Feature Engineering Schematics",
        run_feature_engineering_analysis,
        project_root,
    )

    results["Model Performance"] = _run_section(
        "Model Performance Comparison",
        run_model_performance_analysis,
        project_root,
    )

    results["Calibration"] = _run_section(
        "Calibration / Reliability Diagrams",
        run_calibration_analysis,
        project_root,
    )

    if df_with_xg is not None:
        results["xG Prediction"] = _run_section(
            "xG Predictive Power Analysis",
            run_xg_prediction_analysis,
            project_root,
            df_with_xg,
        )
    else:
        print("\n  Skipping xG Prediction Analysis (no model predictions available).")
        results["xG Prediction"] = False

    # ------------------------------------------------------------------
    # 4. Summary.
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 60)
    print("  Exploration Pipeline Summary")
    print("=" * 60)
    for section, passed in results.items():
        status = "OK" if passed else "FAILED"
        print(f"  {section:<30} {status}")
    print(f"\n  Total time: {total_elapsed:.1f}s")
    print("=" * 60 + "\n")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
