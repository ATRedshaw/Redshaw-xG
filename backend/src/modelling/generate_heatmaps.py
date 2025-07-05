import json
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path for module imports.
# This allows the script to be run directly while finding 'backend' modules.
try:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(project_root))
    from backend.utils.helper import load_models, load_metadata_features, determine_model
    from backend.utils.preprocess import preprocess
except ImportError as e:
    print(f"Error: Could not import project modules. Ensure the script is in the correct directory structure.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration ---
OUTPUT_FILENAME = 'heatmaps/heatmaps.json'
GRID_RESOLUTION = 0.01

def _generate_heatmap_for_combo(
    situation: str | None,
    shot_type: str | None,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    models: dict,
    features: dict
) -> list[list[float]]:
    """
    Generates a single 2D heatmap grid for a specific situation/shot-type combo.
    
    This function isolates the prediction logic for one heatmap, improving modularity.
    It iterates through each point on the grid, determines the correct model,
    preprocesses the input, and gets the xG prediction.
    """
    # Handle the special case for Penalties, which have a fixed xG value.
    if situation == 'Penalty':
        # A penalty's xG is constant and not dependent on minor coordinate variations.
        return (np.full((len(y_coords), len(x_coords)), 0.76)).tolist()

    # Initialize a NumPy array to hold the xG values for the grid.
    # Using NumPy allows for efficient assignment.
    xg_grid = np.zeros((len(y_coords), len(x_coords)))

    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # Determine the appropriate model for the given coordinates and context.
            model_info = determine_model(x, y, situation, shot_type, {'is_normalised': True})
            
            # If no model is applicable (e.g., shot from behind goal line), xG is 0.
            if model_info['error']:
                xg_grid[i, j] = 0.0
                continue

            chosen_model = model_info['chosen_model']
            model_features = features[chosen_model]
            
            # Preprocess the point data for the selected model.
            X = preprocess(x, y, situation, shot_type, chosen_model, model_features)
            
            # Predict the probability and store it, rounded for precision.
            prediction = models[chosen_model].predict_proba(X)[:, 1][0]
            xg_grid[i, j] = round(prediction, 4)

    # Convert the final NumPy grid to a standard Python list for JSON serialization.
    return xg_grid.tolist()

def generate_heatmaps(output_path: str = OUTPUT_FILENAME):
    """
    Generates and saves structured heatmap data for various model combinations.

    Data Structure Rationale:
    The output JSON is structured to be efficient and easy to parse on the client-side.
    - `grid_definition`: Defines the x and y axes for all heatmaps. This avoids
      repeating coordinate data for every heatmap, significantly reducing file size.
    - `heatmaps`: A nested dictionary (`situation -> shot_type -> grid`). This
      hierarchical structure is more intuitive and allows for direct lookups
      (e.g., `data.heatmaps['OpenPlay']['Head']`) without string manipulation.
    - The actual heatmap is a 2D array of xG values, which directly maps to the
      grid defined in `grid_definition`.
    """
    print("Loading models and feature metadata...")
    models = load_models(path='models')
    features = load_metadata_features(path='models')

    # Define the discrete categories for which heatmaps will be generated.
    # `None` represents an 'any' or 'all-encompassing' category.
    situations = [None, 'OpenPlay', 'SetPiece', 'DirectFreekick', 'FromCorner']
    shot_types = [None, 'Head', 'RightFoot', 'LeftFoot', 'OtherBodyPart']
    
    # Generate the coordinate grid. This is done once and reused for all heatmaps.
    x_coords = np.round(np.arange(0, 1.0 + GRID_RESOLUTION, GRID_RESOLUTION), 2)
    y_coords = np.round(np.arange(0, 1.0 + GRID_RESOLUTION, GRID_RESOLUTION), 2)
    
    # --- Data Assembly ---
    # This dictionary will hold the final, structured data.
    structured_data = {
        # Store grid coordinates once to reduce file size and redundancy.
        'grid_definition': {
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist()
        },
        # Use a nested dictionary for intuitive, hierarchical access to heatmaps.
        'heatmaps': {}
    }

    # Use tqdm for a clear progress indicator during the generation process.
    for situation in tqdm(situations, desc='Situations'):
        # Use string 'None' as key for JSON compatibility and clarity.
        situation_key = situation if situation is not None else 'None'
        structured_data['heatmaps'][situation_key] = {}

        for shot_type in tqdm(shot_types, desc='Shot Types', leave=False):
            shot_type_key = shot_type if shot_type is not None else 'None'
            
            # Generate the 2D grid of xG values for the current combination.
            heatmap_grid = _generate_heatmap_for_combo(
                situation, shot_type, x_coords, y_coords, models, features
            )
            
            structured_data['heatmaps'][situation_key][shot_type_key] = heatmap_grid

    # --- File Output ---
    output_file = Path(output_path)
    # Ensure the parent directory exists before attempting to write the file.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving structured heatmap data to {output_file}...")
    with open(output_file, 'w') as f:
        # Dump the structured data to JSON. No indent for smaller file size.
        json.dump(structured_data, f)

    print("Heatmap data generation complete.")

if __name__ == '__main__':
    generate_heatmaps()