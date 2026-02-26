"""Exports all scikit-learn joblib models to ONNX format for client-side inference.

Reads each model pipeline from ./models/{name}/model.joblib, converts it to ONNX,
and writes the output alongside a copied metadata.json to ../frontend/models/{name}/.
Also copies heatmaps/heatmaps.json to ../frontend/data/heatmaps.json so the static
frontend has no dependency on a live API endpoint.

Usage (run from the backend/ directory):
    pip install skl2onnx onnx
    python export_to_onnx.py
"""

import json
import shutil
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError as exc:
    raise ImportError(
        "skl2onnx is required. Install it with: pip install skl2onnx onnx"
    ) from exc

# Paths are relative to the backend/ directory where this script is executed.
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("../frontend/models")


def export_model(model_name: str, model_dir: Path) -> None:
    """Load a scikit-learn pipeline from joblib and convert it to ONNX.

    The resulting .onnx file and a copy of metadata.json are placed in
    OUTPUT_DIR/{model_name}/.

    Args:
        model_name: Subdirectory name for the model (e.g., 'basic_model').
        model_dir: Path to the model's source directory.
    """
    joblib_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    if not joblib_path.exists():
        print(f"[SKIP] No model.joblib in {model_dir}")
        return

    if not metadata_path.exists():
        print(f"[SKIP] No metadata.json in {model_dir}")
        return

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    n_features: int = len(metadata["features"])
    model = joblib.load(joblib_path)

    # FloatTensorType ensures the ONNX graph expects float32 input, matching
    # the Float32Array supplied by ONNX Runtime Web in the browser.
    initial_types = [("float_input", FloatTensorType([None, n_features]))]

    # Disabling zipmap converts the 'probabilities' output from a sequence of
    # maps to a plain float32 tensor of shape [N, 2], which ORT Web can consume
    # directly without additional post-processing.
    options = {LogisticRegression: {"zipmap": False}}

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options=options,
        target_opset=17,
    )

    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    shutil.copy(metadata_path, output_dir / "metadata.json")
    print(f"[OK]  {model_name:<22} features={n_features:2d}  ->  {onnx_path}")


def copy_heatmaps() -> None:
    """Copy the pre-generated heatmaps.json to the frontend data directory.

    This makes the heatmap data available as a static asset, removing the
    need for a live API call.
    """
    src = Path("heatmaps/heatmaps.json")
    dst = Path("../frontend/data/heatmaps.json")

    if not src.exists():
        print(f"\n[WARN] heatmaps.json not found at {src} — skipping copy.")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"\n[OK]  Copied heatmaps.json  ->  {dst}")


def main() -> None:
    """Iterate all model subdirectories, export each to ONNX, then copy heatmaps."""
    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            f"Models directory '{MODELS_DIR}' not found. "
            "Run this script from the backend/ directory."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(d for d in MODELS_DIR.iterdir() if d.is_dir())
    if not model_dirs:
        print("No model subdirectories found.")
        return

    print("Exporting models to ONNX...\n")
    for model_dir in model_dirs:
        export_model(model_dir.name, model_dir)

    copy_heatmaps()
    print("\nExport complete.")


if __name__ == "__main__":
    main()
