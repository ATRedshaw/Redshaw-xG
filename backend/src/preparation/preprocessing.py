import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path
import os
from datetime import datetime

# Define constants
GOAL_CENTER = (1.0, 0.5)  # (X, Y) coordinates of the goal center
GOAL_POSTS = [(1.0, 0.45), (1.0, 0.55)]  # (X, Y) coordinates of goal posts
WIDE_AREA_THRESHOLD = 0.2  # Threshold for defining wide areas (Y < 0.2 or Y > 0.8)


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the input DataFrame for xG modeling.
    
    Args:
        df: Input DataFrame containing raw features
            Required columns: 'X', 'Y', 'situation', 'shotType', 'result'
            
    Returns:
        Tuple containing:
            - DataFrame with all engineered features
            - Series with the target variable (1 for goal, 0 otherwise)
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # 1. Encode target variable
    y = encode_target(df['result'])
    
    # 2. Create features
    # 2.1 Add original normalized coordinates (will be reordered later)
    df_features = df[['X', 'Y']].copy()
    
    # 2.2 One-hot encode categorical features
    situation_dummies = pd.get_dummies(df['situation'], prefix='situation')
    shot_type_dummies = pd.get_dummies(df['shotType'], prefix='shotType')
    
    # 2.3 Calculate numerical features
    df_features['distance_to_goal'] = calculate_distance_to_goal(df[['X', 'Y']])
    df_features['angle_to_goal'] = calculate_angle_to_goal(df[['X', 'Y']])
    
    # 2.4 Create interaction terms
    interaction_dummies = create_interaction_terms(df[['situation', 'shotType']])
    
    # Combine all features with logical ordering
    X = pd.concat([
        df_features,                    # X, Y coordinates
        situation_dummies,              # Situation types
        shot_type_dummies,              # Shot types
        interaction_dummies,            # Interaction terms
        y.rename('target')              # Target variable (will be moved later)
    ], axis=1)
    
    # Reorder columns to put target at the end
    cols = X.columns.tolist()
    if 'target' in cols:
        cols.remove('target')
        cols.append('target')
    X = X[cols]
    
    # Separate features and target again
    y = X.pop('target')
    
    return X, y, cols[:-1]  # Return features, target, and feature names

def encode_target(result_series: pd.Series) -> pd.Series:
    """Encode the target variable (1 for 'Goal', 0 otherwise)."""
    return (result_series == 'Goal').astype(int)

def calculate_distance_to_goal(coords: pd.DataFrame) -> pd.Series:
    """Calculate Euclidean distance from (X,Y) to goal center."""
    return np.sqrt(
        (coords['X'] - GOAL_CENTER[0])**2 + 
        (coords['Y'] - GOAL_CENTER[1])**2
    )

def calculate_angle_to_goal(coords: pd.DataFrame) -> pd.Series:
    """
    Calculate shooting angle in radians using vectors to goalposts.
    
    Args:
        coords: DataFrame with 'X' and 'Y' columns
        
    Returns:
        Series with angles in radians
    """
    # Vectors to goal posts
    v1_x = GOAL_POSTS[0][0] - coords['X']
    v1_y = GOAL_POSTS[0][1] - coords['Y']
    v2_x = GOAL_POSTS[1][0] - coords['X']
    v2_y = GOAL_POSTS[1][1] - coords['Y']
    
    # Dot product and magnitudes
    dot_product = v1_x * v2_x + v1_y * v2_y
    mag_v1 = np.sqrt(v1_x**2 + v1_y**2)
    mag_v2 = np.sqrt(v2_x**2 + v2_y**2)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_angle = dot_product / (mag_v1 * mag_v2)
        # Handle potential numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
    
    return angle

def create_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction terms between situation and shotType."""
    # Create interaction column
    interaction = df['situation'] + '_' + df['shotType']
    
    # One-hot encode the interaction terms
    return pd.get_dummies(interaction, prefix='interaction')

def save_preprocessed_data(X: pd.DataFrame, y: pd.Series, output_dir: Path, 
                         filename: str = 'preprocessed_shots.csv') -> Path:
    """
    Save the preprocessed data to a CSV file.
    
    Args:
        X: Features DataFrame
        y: Target Series
        output_dir: Directory to save the output file
        filename: Output filename
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the provided filename as-is
    output_path = output_dir / filename
    
    # Combine features and target
    data_with_target = X.copy()
    data_with_target['target'] = y
    
    # Save to CSV
    data_with_target.to_csv(output_path, index=False)
    print(f"\nPreprocessed data saved to: {output_path}")
    
    # Also save column order to a text file
    with open(output_dir / 'feature_columns.txt', 'w') as f:
        f.write('\n'.join(X.columns.tolist() + ['target']))
    
    return output_path

def main():
    try:
        # Construct paths
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / 'data' / 'cleansed' / 'all_shots_cleansed.csv'
        output_dir = project_root / 'data' / 'preprocessed'
        
        print(f"Reading data from: {data_path}")
        
        # Read the data
        df = pd.read_csv(data_path)
        
        # Display basic info about the data
        print("\nOriginal data shape:", df.shape)
        print("\nFirst few rows of original data:")
        print(df.head())
        
        # Check for required columns
        required_columns = ['X', 'Y', 'situation', 'shotType', 'result']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Preprocess the data
        print("\nPreprocessing data...")
        X, y, feature_names = preprocess_data(df)
        
        # Display information about the preprocessed data
        print("\nPreprocessing complete!")
        print("\nFeatures shape:", X.shape)
        print("Target shape:", y.shape)
        
        # Print column order
        print("\nFeature columns (in order):")
        for i, col in enumerate(X.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Display class distribution
        print("\nClass distribution in target:")
        print(y.value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        
        # Save preprocessed data
        save_preprocessed_data(X, y, output_dir)
        
        return X, y
    
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()