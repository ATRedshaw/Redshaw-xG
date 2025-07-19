import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from typing import List, Tuple
import warnings
from tqdm import tqdm

# --- Constants ---
GOAL_CENTER = (1.0, 0.5)
GOAL_POSTS = [(1.0, 0.45), (1.0, 0.55)]

# --- Data Loading and Preparation ---

def get_raw_data_files(data_dir: Path) -> List[Path]:
    """
    Scans a directory recursively for all raw shot data files ('shots_*.csv').

    Args:
        data_dir: The root directory for raw data (e.g., '.../data/raw').

    Returns:
        A list of Path objects for all found shot data files.
    """
    print("Searching for raw data files...")
    files = list(data_dir.glob('**/*shots_*.csv'))
    print(f"Found {len(files)} files to process.")
    return files

def load_and_combine_raw_data(files: List[Path]) -> pd.DataFrame:
    """
    Loads multiple raw shot data CSVs into a single pandas DataFrame.

    Args:
        files: A list of file paths to load.

    Returns:
        A combined DataFrame of all shot data, or an empty DataFrame if loading fails.
    """
    if not files:
        print("No files provided to load. Returning empty DataFrame.")
        return pd.DataFrame()
        
    all_shots = []
    for file_path in tqdm(files, desc="Loading raw data files"):
        try:
            df = pd.read_csv(file_path)
            all_shots.append(df)
        except Exception as e:
            # Log a warning but continue processing other files.
            print(f"Warning: Could not process {file_path}: {e}")
            
    if not all_shots:
        print("No data was successfully loaded. Exiting.")
        return pd.DataFrame()

    combined_df = pd.concat(all_shots, ignore_index=True)
    print(f"Successfully combined {len(files)} files into a DataFrame with {len(combined_df):,} rows.")
    return combined_df

# --- Feature Engineering ---

def calculate_distance_to_goal(coords: pd.DataFrame) -> pd.Series:
    """Calculates the Euclidean distance from a shot's (X, Y) to the goal center."""
    return np.sqrt((coords['X'] - GOAL_CENTER[0])**2 + (coords['Y'] - GOAL_CENTER[1])**2)

def calculate_angle_to_goal(coords: pd.DataFrame) -> pd.Series:
    """Calculates the shooting angle in radians using vectors to the goalposts."""
    # Define vectors from the shot location to each goalpost.
    v1_x = GOAL_POSTS[0][0] - coords['X']
    v1_y = GOAL_POSTS[0][1] - coords['Y']
    v2_x = GOAL_POSTS[1][0] - coords['X']
    v2_y = GOAL_POSTS[1][1] - coords['Y']
    
    # Calculate the dot product and magnitudes of the vectors.
    dot_product = v1_x * v2_x + v1_y * v2_y
    mag_v1 = np.sqrt(v1_x**2 + v1_y**2)
    mag_v2 = np.sqrt(v2_x**2 + v2_y**2)
    
    # Use arccos to find the angle, handling potential floating point errors.
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0) # Ensure the value is within arccos's domain.
        angle = np.arccos(cos_angle)
    
    return angle

def create_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Creates one-hot encoded interaction features between situation and shotType."""
    interaction = df['situation'] + '_' + df['shotType']
    return pd.get_dummies(interaction, prefix='interaction')

def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning and feature engineering to raw shot data to prepare it for prediction.

    Args:
        df: The raw, combined DataFrame of shot data.
        
    Returns:
        A DataFrame with all features required for any of the trained models.
    """
    print("Preprocessing data for prediction...")
    df_clean = df.copy()
    
    # 1. Enforce correct data types and handle missing values.
    required_cols = ['X', 'Y', 'situation', 'shotType']
    for col in required_cols:
        if col not in df_clean.columns:
            raise ValueError(f"Raw data is missing required column: {col}")
            
    df_clean['X'] = pd.to_numeric(df_clean['X'], errors='coerce')
    df_clean['Y'] = pd.to_numeric(df_clean['Y'], errors='coerce')
    df_clean['situation'] = df_clean['situation'].astype(str)
    df_clean['shotType'] = df_clean['shotType'].astype(str)
    
    # Drop records where coordinates could not be parsed, as they are essential.
    df_clean.dropna(subset=['X', 'Y'], inplace=True)
    
    # 2. Engineer all required features.
    # Base distance and angle features.
    features = pd.DataFrame(index=df_clean.index)
    features['X'] = df_clean['X']
    features['Y'] = df_clean['Y']
    features['distance_to_goal'] = calculate_distance_to_goal(df_clean[['X', 'Y']])
    features['angle_to_goal'] = calculate_angle_to_goal(df_clean[['X', 'Y']])
    
    # One-hot encoded categorical and interaction features.
    situation_dummies = pd.get_dummies(df_clean['situation'], prefix='situation')
    shottype_dummies = pd.get_dummies(df_clean['shotType'], prefix='shotType')
    interaction_dummies = create_interaction_terms(df_clean[['situation', 'shotType']])
    
    # Combine all feature sets into a single DataFrame.
    all_features = pd.concat([
        features,
        situation_dummies,
        shottype_dummies,
        interaction_dummies
    ], axis=1)
    
    print("Preprocessing complete.")
    return all_features


# --- Model Prediction ---

def apply_models_and_get_xg(data_raw: pd.DataFrame, features_df: pd.DataFrame, models_dir: Path) -> pd.DataFrame:
    """
    Loads trained models, generates xG predictions, and appends them to the DataFrame.

    Args:
        data_raw: The original DataFrame (aligned with features_df).
        features_df: DataFrame with all engineered features.
        models_dir: The directory where model artifacts are stored.

    Returns:
        The original DataFrame with new xG columns appended for each model.
    """
    model_names = ['basic_model', 'situation_model', 'shottype_model', 'advanced_model']
    output_df = data_raw.loc[features_df.index].copy() # Ensure alignment after dropna.

    for model_name in model_names:
        print(f"--- Applying {model_name} ---")
        model_path = models_dir / model_name / "model.joblib"
        metadata_path = models_dir / model_name / "metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            print(f"Warning: Artifacts not found for '{model_name}'. Skipping.")
            continue
            
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_features = metadata['features']
        
        # Align the feature DataFrame with the model's expected input.
        # This ensures all required columns are present and in the correct order,
        # filling any missing columns (e.g., rare categories) with 0.
        features_aligned = features_df.reindex(columns=model_features, fill_value=0)

        # Print input columns for the first row of shots
        print(f"Input columns for {model_name} (first row):")
        print(features_aligned.iloc[0].to_dict())
        
        # Suppress sklearn's feature name warning during prediction.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Predict probabilities and select the probability of the positive class (Goal).
            xg_predictions = model.predict_proba(features_aligned)[:, 1]

        output_df[f'xG_{model_name}'] = xg_predictions
        print(f"Successfully generated and appended 'xG_{model_name}'.")

    return output_df


def main():
    """
    Main execution block to run the full data processing and prediction pipeline.
    This can be used for direct script execution and testing.
    """
    # Define project paths. This assumes the script is run from its location in
    # 'exploration/preprocessing/preprocess.py'.
    try:
        project_root = Path(__file__).parent.parent.parent
    except NameError:
        # Fallback for interactive environments (like Jupyter).
        project_root = Path('.').resolve().parent.parent
        
    raw_data_dir = project_root / 'data' / 'raw'
    models_dir = project_root / 'models'

    # 1. Load and combine all raw shot data.
    raw_files = get_raw_data_files(raw_data_dir)
    if not raw_files:
        print("No raw data files found. Exiting.")
        return
    
    df_raw_combined = load_and_combine_raw_data(raw_files)
    if df_raw_combined.empty:
        return
        
    # 2. Create the feature set for prediction.
    df_features = preprocess_for_prediction(df_raw_combined)
    
    # 3. Apply models to generate xG values.
    df_with_xg = apply_models_and_get_xg(df_raw_combined, df_features, models_dir)
    
    # 4. Display a summary of the results.
    print("\n--- Exploration Complete ---")
    print("DataFrame with new xG columns (first 5 rows):")
    
    display_cols = [
        'player', 'minute', 'result', 'situation', 'shotType', 
        'xG_basic_model', 'xG_situation_model', 'xG_shottype_model', 'xG_advanced_model'
    ]
    # Ensure we only try to display columns that exist.
    display_cols = [col for col in display_cols if col in df_with_xg.columns]

    print(df_with_xg[display_cols].head())
    print(f"\nFinal DataFrame shape: {df_with_xg.shape}")
    
    return df_with_xg

if __name__ == '__main__':
    final_df = main()