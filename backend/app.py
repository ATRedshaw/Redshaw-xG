from utils.helper import return_model_paths
import os
import joblib
import pandas as pd
import json

def load_models_from_paths(models):
    """Load models from the specified paths using joblib."""
    loaded_models = {}
    for model_name, path in models.items():
        model_path = os.path.join(path, 'model.pkl')
        if os.path.exists(model_path):
            try:
                # Load the model using joblib
                loaded_models[model_name] = joblib.load(model_path)
                print(f"Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
        else:
            print(f"Model file not found for {model_name} at {model_path}")
    return loaded_models

if __name__ == "__main__":
    # Get model paths
    model_paths = return_model_paths()
    
    # Load all models
    loaded_models = load_models_from_paths(model_paths)

    test_col_values = [
        "X",
        "Y",
        "distance_to_goal",
        "angle_to_goal",
        "zone_central",
        "zone_wide"
    ]

    test_data = pd.DataFrame({
        "X": [0.5],
        "Y": [0.5],
        "distance_to_goal": [0.5],
        "angle_to_goal": [0.1],
        "zone_central": [1],
        "zone_wide": [0]
    })

    # Test basic_model-xg
    basic_model = loaded_models['basic_model-xg']
    result = basic_model.predict_proba(test_data)[:,1]
    print(result)

