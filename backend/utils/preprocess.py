import numpy as np
import pandas as pd

# Define constants
GOAL_CENTER = (1.0, 0.5)  # (X, Y) coordinates of the goal center
GOAL_POSTS = [(1.0, 0.45), (1.0, 0.55)]  # (X, Y) coordinates of goal posts
WIDE_AREA_THRESHOLD = 0.2  # Threshold for defining wide areas (Y < 0.2 or Y > 0.8)

def preprocess(x, y, situation, shot_type, chosen_model, chosen_model_features):
    """Preprocess the data to be used by the models for inference.
    
    Args:
        x: x-coordinate of the shot (assumed validated and normalized).
        y: y-coordinate of the shot (assumed validated and normalized).
        situation: The situation of the shot (e.g., 'OpenPlay', 'SetPiece') or None.
        shot_type: The type of the shot (e.g., 'Head', 'RightFoot') or None.
        chosen_model: The selected model ('basic_model', 'shottype_model', 'situation_model', 'advanced_model').
        chosen_model_features: List of feature names required by the chosen model.
    
    Returns:
        pandas.DataFrame containing the preprocessed features for the chosen model with a single row.
    
    Raises:
        ValueError: If chosen_model is invalid or chosen_model_features is empty.
    """
    # Validate inputs
    if not isinstance(chosen_model_features, list) or not chosen_model_features:
        raise ValueError("chosen_model_features must be a non-empty list")
    if chosen_model not in ['basic_model', 'shottype_model', 'situation_model', 'advanced_model']:
        raise ValueError(f"Invalid chosen_model: {chosen_model}")

    # Helper functions to compute and add features
    def calculate_distance_to_goal(features, x, y):
        """Calculate Euclidean distance from (x, y) to goal center."""
        if 'distance_to_goal' in features:
            features['distance_to_goal'] = np.sqrt(
                (x - GOAL_CENTER[0])**2 + 
                (y - GOAL_CENTER[1])**2
            )
        return features

    def calculate_angle_to_goal(features, x, y):
        """Calculate shooting angle in radians using vectors to goalposts."""
        if 'angle_to_goal' in features:
            # Vectors to goal posts
            v1_x = GOAL_POSTS[0][0] - x
            v1_y = GOAL_POSTS[0][1] - y
            v2_x = GOAL_POSTS[1][0] - x
            v2_y = GOAL_POSTS[1][1] - y
            
            # Dot product and magnitudes
            dot_product = v1_x * v2_x + v1_y * v2_y
            mag_v1 = np.sqrt(v1_x**2 + v1_y**2)
            mag_v2 = np.sqrt(v2_x**2 + v2_y**2)
            
            # Handle division by zero or invalid values
            if mag_v1 == 0 or mag_v2 == 0:
                features['angle_to_goal'] = 0
            else:
                cos_angle = dot_product / (mag_v1 * mag_v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                features['angle_to_goal'] = np.arccos(cos_angle)
        return features

    def add_situation_features(features, situation):
        """Add one-hot encoded situation features."""
        if situation is not None:
            situation_feat = f'situation_{situation}'
            if situation_feat in features:
                features[situation_feat] = 1
        return features

    def add_shot_type_features(features, shot_type):
        """Add one-hot encoded shot type features."""
        if shot_type is not None:
            shot_type_feat = f'shotType_{shot_type}'
            if shot_type_feat in features:
                features[shot_type_feat] = 1
        return features

    def add_interaction_features(features, situation, shot_type):
        """Add one-hot encoded interaction features between situation and shot_type."""
        if situation is not None and shot_type is not None:
            interaction_feat = f'interaction_{situation}_{shot_type}'
            if interaction_feat in features:
                features[interaction_feat] = 1
        return features

    def reorder_features(features, chosen_model_features):
        """Ensure all required features are present and ordered correctly."""
        missing_features = [feat for feat in chosen_model_features if feat not in features]
        if missing_features:
            raise ValueError(f"Missing features in output: {missing_features}")
        return features

    # Initialize features dictionary with all required features set to 0
    features = {feat: 0 for feat in chosen_model_features}

    # Set coordinate features if required
    if 'X' in chosen_model_features:
        features['X'] = float(x) if x is not None else 0.0
    if 'Y' in chosen_model_features:
        features['Y'] = float(y) if y is not None else 0.0

    # Calculate numerical features
    features = calculate_distance_to_goal(features, x, y)
    features = calculate_angle_to_goal(features, x, y)

    # Add categorical and interaction features based on the chosen model
    if chosen_model in ['situation_model', 'advanced_model']:
        features = add_situation_features(features, situation)
    
    if chosen_model in ['shottype_model', 'advanced_model']:
        features = add_shot_type_features(features, shot_type)
    
    if chosen_model == 'advanced_model':
        features = add_interaction_features(features, situation, shot_type)

    # Validate and reorder features
    features = reorder_features(features, chosen_model_features)

    # Convert to DataFrame with columns in the order specified by chosen_model_features
    features_df = pd.DataFrame([features], columns=chosen_model_features)

    return features_df

if __name__ == '__main__':
    # Define feature order for each model
    model_features = {
        'basic_model': ['X', 'Y', 'distance_to_goal', 'angle_to_goal'],
        'shottype_model': ['X', 'Y', 'distance_to_goal', 'angle_to_goal', 'shotType_Head', 'shotType_LeftFoot', 'shotType_OtherBodyPart', 'shotType_RightFoot'],
        'situation_model': ['X', 'Y', 'distance_to_goal', 'angle_to_goal', 'situation_DirectFreekick', 'situation_FromCorner', 'situation_OpenPlay', 'situation_Penalty', 'situation_SetPiece'],
        'advanced_model': ['X', 'Y', 'distance_to_goal', 'angle_to_goal', 'situation_DirectFreekick', 'situation_FromCorner', 'situation_OpenPlay', 'situation_Penalty', 'situation_SetPiece', 'shotType_Head', 'shotType_LeftFoot', 'shotType_OtherBodyPart', 'shotType_RightFoot', 'interaction_DirectFreekick_LeftFoot', 'interaction_DirectFreekick_RightFoot', 'interaction_FromCorner_Head', 'interaction_FromCorner_LeftFoot', 'interaction_FromCorner_OtherBodyPart', 'interaction_FromCorner_RightFoot', 'interaction_OpenPlay_Head', 'interaction_OpenPlay_LeftFoot', 'interaction_OpenPlay_OtherBodyPart', 'interaction_OpenPlay_RightFoot', 'interaction_Penalty_LeftFoot', 'interaction_Penalty_RightFoot', 'interaction_SetPiece_Head', 'interaction_SetPiece_LeftFoot', 'interaction_SetPiece_OtherBodyPart', 'interaction_SetPiece_RightFoot']
    }

    # Define multiple test cases
    test_cases = [
        # Test Case 1: Original basic_model test case (central shot)
        {
            'name': 'Basic Model - Central Shot',
            'x': 0.5,
            'y': 0.5,
            'situation': None,
            'shot_type': None,
            'chosen_model': 'basic_model',
            'chosen_model_features': model_features['basic_model']
        },
        # Test Case 2: Basic Model - Wide shot (Y < 0.2)
        {
            'name': 'Basic Model - Wide Shot',
            'x': 0.8,
            'y': 0.1,
            'situation': None,
            'shot_type': None,
            'chosen_model': 'basic_model',
            'chosen_model_features': model_features['basic_model']
        },
        # Test Case 3: Situation Model - OpenPlay shot
        {
            'name': 'Situation Model - OpenPlay',
            'x': 0.6,
            'y': 0.4,
            'situation': 'OpenPlay',
            'shot_type': None,
            'chosen_model': 'situation_model',
            'chosen_model_features': model_features['situation_model']
        },
        # Test Case 4: ShotType Model - RightFoot shot
        {
            'name': 'ShotType Model - RightFoot',
            'x': 0.7,
            'y': 0.6,
            'situation': None,
            'shot_type': 'RightFoot',
            'chosen_model': 'shottype_model',
            'chosen_model_features': model_features['shottype_model']
        },
        # Test Case 5: Advanced Model - Set Piece with Right Foot
        {
            'name': 'Advanced Model - Set Piece Right Foot',
            'x': 0.9,
            'y': 0.5,
            'situation': 'SetPiece',
            'shot_type': 'RightFoot',
            'chosen_model': 'advanced_model',
            'chosen_model_features': model_features['advanced_model']
        },
        # Test Case 6: Edge Case - Shot near goalpost (Y = 0.45)
        {
            'name': 'Basic Model - Near Goalpost',
            'x': 0.99,
            'y': 0.45,
            'situation': None,
            'shot_type': None,
            'chosen_model': 'basic_model',
            'chosen_model_features': model_features['basic_model']
        }
    ]

    # Run each test case
    for test in test_cases:
        print(f"\n=== {test['name']} ===")
        try:
            result = preprocess(
                x=test['x'],
                y=test['y'],
                situation=test['situation'],
                shot_type=test['shot_type'],
                chosen_model=test['chosen_model'],
                chosen_model_features=test['chosen_model_features']
            )
            print("Test input:", {k: v for k, v in test.items() if k != 'name'})
            print("Preprocessed features:\n", result)
        except Exception as e:
            print(f"Error in test case '{test['name']}': {str(e)}")