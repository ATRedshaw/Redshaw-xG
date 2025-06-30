import joblib
from pathlib import Path
import json
import pandas as pd

def load_models(path='models'):
    """Load all joblib models from the specified directory into a dictionary.
    
    Args:
        path (str): Directory path containing model folders (default: 'models')
        
    Returns:
        dict: Dictionary with model names as keys and loaded joblib models as values
    """
    models = {}
    models_dir = Path(path)
    
    # Check if directory exists
    if not models_dir.exists():
        raise FileNotFoundError(f"Directory '{path}' not found")
    
    # Iterate through subdirectories
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # Look for .joblib file in the subdirectory
            joblib_file = model_dir / 'model.joblib'
            if joblib_file.exists():
                try:
                    # Load the model and add to dictionary with model name as key
                    model_name = model_dir.name
                    models[model_name] = joblib.load(joblib_file)
                except Exception as e:
                    print(f"Error loading model from {joblib_file}: {str(e)}")
                    
    return models

def load_metadata_features(path='models'):
    """Load 'features' key values from metadata.json files in the specified directory into a dictionary.
    
    Args:
        path (str): Directory path containing model folders with metadata.json files (default: 'models')
        
    Returns:
        dict: Dictionary with model names as keys and 'features' values from metadata.json as values
    """
    features = {}
    models_dir = Path(path)
    
    # Check if directory exists
    if not models_dir.exists():
        raise FileNotFoundError(f"Directory '{path}' not found")
    
    # Iterate through subdirectories
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # Look for metadata.json file in the subdirectory
            metadata_file = model_dir / 'metadata.json'
            if metadata_file.exists():
                try:
                    # Load the metadata and extract 'features' key
                    model_name = model_dir.name
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if 'features' in metadata:
                            features[model_name] = metadata['features']
                except Exception as e:
                    print(f"Error loading metadata from {metadata_file}: {str(e)}")
                    
    return features

def verify_valid_situation(situation):
    """Verify that the situation is valid.
    
    Args:
        situation (str): The situation of the shot (e.g., 'OpenPlay', 'SetPiece').
    
    Returns:
        bool: True if the situation is valid, False otherwise.
    """
    if situation is None:
        return True
    
    if situation in ['OpenPlay', 'SetPiece', 'DirectFreekick', 'FromCorner', 'Penalty']:
        return True
    
    return False

def verify_valid_shot_type(shot_type):
    """Verfiy that the shot type is valid.
    
    Args:
        shot_type (str): The type of the shot (e.g., 'Head', 'RightFoot').
    
    Returns:
        bool: True if the shot type is valid, False otherwise.
    """
    if shot_type is None:
        return True
    
    if shot_type in ['Head', 'RightFoot', 'LeftFoot', 'OtherBodyPart']:
        return True
    
    return False

def verify_all_features_present(X, features):
    """Verify all of the features needed for the model are present

    Args:
        X (pandas.DataFrame): The input data.
        features (list): The list of features needed for the model.
    
    Returns:
        bool: True if all features are present, False otherwise.
    """
    for feature in features:
        if feature not in X.columns:
            return False
    
    return True

def determine_model(x, y, situation, shot_type, normalisation):
    """Determine the appropriate model based on input features.
    
    Args:
        x (float): x-coordinate of the shot.
        y (float): y-coordinate of the shot.
        situation (str): The situation of the shot (e.g., 'OpenPlay', 'SetPiece').
        shot_type (str): The type of the shot (e.g., 'Head', 'RightFoot').
        normalisation (dict): The normalisation parameters.
    
    Returns:
        dict: A dictionary containing the chosen model, x and y coordinates, and error message.
    """
    # Validate x or y presence and numeric type
    if x is None or y is None:
        return {
            'chosen_model': None,
            'x': None,
            'y': None,
            'error': "Missing x or y coordinates - Coordinates must be included to create a prediction."
        }
    
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return {
            'chosen_model': None,
            'x': None,
            'y': None,
            'error': "x and y must be numeric values."
        }

    # Validate normalisation dictionary
    if not isinstance(normalisation, dict) or 'is_normalised' not in normalisation:
        return {
            'chosen_model': None,
            'x': None,
            'y': None,
            'error': "Normalisation dictionary must contain 'is_normalised' key."
        }

    # Handle normalisation if needed
    if normalisation['is_normalised'] == False:
        max_width = normalisation.get('max_pitch_width')
        max_length = normalisation.get('max_pitch_length')

        if max_width is None or max_length is None:
            return {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "In order to carry out normalisation for feature engineering and model inference a max pitch width and a max pitch length need to be provided."
            }
        
        try:
            max_width, max_length = float(max_width), float(max_length)
            if max_width <= 0 or max_length <= 0:
                return {
                    'chosen_model': None,
                    'x': None,
                    'y': None,
                    'error': "The maximum width and length of the pitch cannot be a negative value."
                }
            x = x / max_width
            y = y / max_length
        except (TypeError, ValueError):
            return {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "max_pitch_width and max_pitch_length must be positive numbers."
            }

    # Validate coordinate range
    if x < 0 or x > 1 or y < 0 or y > 1:
        return {
            'chosen_model': None,
            'x': None,
            'y': None,
            'error': "Coordinates have been incorrectly claimed as normalised - Set is_normalised to false or normalise the coordinates so they are between 0 and 1."
        }

    # Determine model based on input combination
    if situation is None and shot_type is None:
        return {
            'chosen_model': 'basic_model',
            'x': x,
            'y': y,
            'error': None
        }
    elif situation is None and shot_type is not None:
        return {
            'chosen_model': 'shottype_model',
            'x': x,
            'y': y,
            'error': None
        }
    elif situation is not None and shot_type is None:
        return {
            'chosen_model': 'situation_model',
            'x': x,
            'y': y,
            'error': None
        }
    elif situation is not None and shot_type is not None:
        return {
            'chosen_model': 'advanced_model',
            'x': x,
            'y': y,
            'error': None
        }
    else:
        return {
            'chosen_model': None,
            'x': None,
            'y': None,
            'error': "Invalid input combination."
        }
    


if __name__ == '__main__':
    models = load_models()
    print(models)
    inputs = load_metadata_features()
    print(inputs, "\n")

    # Test cases for determine_model function
    test_cases = [
        # Test 1: Valid basic_model with normalised coordinates
        {
            'x': 0.5,
            'y': 0.3,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': 'True'},
            'expected': {
                'chosen_model': 'basic_model',
                'x': 0.5,
                'y': 0.3,
                'error': None
            },
            'description': "Valid basic_model with normalised coordinates"
        },
        # Test 2: Valid basic_model with unnormalised coordinates
        {
            'x': 34.0,
            'y': 52.5,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': False, 'max_pitch_width': 68.0, 'max_pitch_length': 105.0},
            'expected': {
                'chosen_model': 'basic_model',
                'x': 0.5,
                'y': 0.5,
                'error': None
            },
            'description': "Valid basic_model with unnormalised coordinates"
        },
        # Test 3: Valid shottype_model with normalised coordinates
        {
            'x': 0.7,
            'y': 0.2,
            'situation': None,
            'shot_type': 'RightFoot',
            'normalisation': {'is_normalised': True},
            'expected': {
                'chosen_model': 'shottype_model',
                'x': 0.7,
                'y': 0.2,
                'error': None
            },
            'description': "Valid shottype_model with normalised coordinates"
        },
        # Test 4: Valid situation_model with unnormalised coordinates
        {
            'x': 20.4,
            'y': 31.5,
            'situation': 'OpenPlay',
            'shot_type': None,
            'normalisation': {'is_normalised': False, 'max_pitch_width': 68.0, 'max_pitch_length': 105.0},
            'expected': {
                'chosen_model': 'situation_model',
                'x': 0.3,
                'y': 0.3,
                'error': None
            },
            'description': "Valid situation_model with unnormalised coordinates"
        },
        # Test 5: Valid advanced_model with normalised coordinates
        {
            'x': 0.8,
            'y': 0.4,
            'situation': 'Penalty',
            'shot_type': 'LeftFoot',
            'normalisation': {'is_normalised': True},
            'expected': {
                'chosen_model': 'advanced_model',
                'x': 0.8,
                'y': 0.4,
                'error': None
            },
            'description': "Valid advanced_model with normalised coordinates"
        },
        # Test 6: Missing x coordinate
        {
            'x': None,
            'y': 0.3,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': True},
            'expected': {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "Missing x or y coordinates - Coordinates must be included to create a prediction."
            },
            'description': "Missing x coordinate"
        },
        # Test 7: Non-numeric x coordinate
        {
            'x': "invalid",
            'y': 0.3,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': True},
            'expected': {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "x and y must be numeric values."
            },
            'description': "Non-numeric x coordinate"
        },
        # Test 8: Invalid normalisation dictionary
        {
            'x': 0.5,
            'y': 0.3,
            'situation': None,
            'shot_type': None,
            'normalisation': {},
            'expected': {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "Normalisation dictionary must contain 'is_normalised' key."
            },
            'description': "Invalid normalisation dictionary"
        },
        # Test 9: Unnormalised coordinates without max_pitch_width
        {
            'x': 34.0,
            'y': 52.5,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': False, 'max_pitch_length': 105.0},
            'expected': {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "In order to carry out normalisation for feature engineering and model inference a max pitch width and a max pitch length need to be provided."
            },
            'description': "Unnormalised coordinates without max_pitch_width"
        },
        # Test 10: Negative max_pitch_width
        {
            'x': 34.0,
            'y': 52.5,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': False, 'max_pitch_width': -68.0, 'max_pitch_length': 105.0},
            'expected': {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "The maximum width and length of the pitch cannot be a negative value."
            },
            'description': "Negative max_pitch_width"
        },
        # Test 11: Non-numeric max_pitch_length
        {
            'x': 34.0,
            'y': 52.5,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': False, 'max_pitch_width': 68.0, 'max_pitch_length': "invalid"},
            'expected': {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "max_pitch_width and max_pitch_length must be positive numbers."
            },
            'description': "Non-numeric max_pitch_length"
        },
        # Test 12: Coordinates out of range with is_normalised=True
        {
            'x': 70.0,
            'y': 110.0,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': True},
            'expected': {
                'chosen_model': None,
                'x': None,
                'y': None,
                'error': "Coordinates have been incorrectly claimed as normalised - Set is_normalised to false or normalise the coordinates so they are between 0 and 1."
            },
            'description': "Coordinates out of range with is_normalised=True"
        },
        # Test 13: Edge case - x and y at boundaries
        {
            'x': 0.0,
            'y': 1.0,
            'situation': None,
            'shot_type': None,
            'normalisation': {'is_normalised': True},
            'expected': {
                'chosen_model': 'basic_model',
                'x': 0.0,
                'y': 1.0,
                'error': None
            },
            'description': "Edge case - x and y at boundaries"
        }
    ]

    # Run tests
    for i, test in enumerate(test_cases, 1):
        result = determine_model(
            x=test['x'],
            y=test['y'],
            situation=test['situation'],
            shot_type=test['shot_type'],
            normalisation=test['normalisation']
        )
        passed = result == test['expected']
        print(f"Test {i}: {test['description']}")
        print(f"Input: x={test['x']}, y={test['y']}, situation={test['situation']}, shot_type={test['shot_type']}, normalisation={test['normalisation']}")
        print(f"Expected: {test['expected']}")
        print(f"Got: {result}")
        print(f"Passed: {passed}\n")