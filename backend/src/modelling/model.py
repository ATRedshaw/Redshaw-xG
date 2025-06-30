import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import joblib
import json

def train_and_save_model(model_name: str, features: list, data: pd.DataFrame, target_col: str, models_dir: Path):
    """
    Trains a RandomForestClassifier, and saves the model, an input example,
    and metadata to a dedicated local directory.

    Args:
        model_name (str): A descriptive name for the model (e.g., 'basic_model'),
                          which will also be the folder name.
        features (list): The list of feature column names to use for training.
        data (pd.DataFrame): The full dataframe containing features and the target.
        target_col (str): The name of the target variable column.
        models_dir (Path): The root directory where model artifacts will be saved.
    """
    print(f"--- Starting training for: {model_name} ---")
    
    # Create a specific directory for this model's artifacts.
    model_output_dir = models_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts will be saved to: {model_output_dir}")

    # Data is prepared for training.
    X = data[features]
    y = data[target_col]

    # Hyperparameter search space.
    search_spaces = {
        'n_estimators': Integer(100, 500),
        'max_depth': Integer(10, 40),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2'])
    }

    # BayesSearchCV is used for hyperparameter optimisation.
    rf = RandomForestClassifier(random_state=104)
    bayes_search = BayesSearchCV(
        estimator=rf,
        search_spaces=search_spaces,
        n_iter=15,
        cv=5,
        scoring='neg_brier_score',
        n_jobs=-1,
        random_state=104,
        verbose=1,
        refit=True
    )

    # Cross-validation is run and the model is fitted using the defined search.
    print(f"Running BayesSearchCV for {model_name}...")
    bayes_search.fit(X, y)

    # The best estimator is trained on the full data because of `refit=True`.
    best_model = bayes_search.best_estimator_
    best_score = -bayes_search.best_score_
    best_params = bayes_search.best_params_

    print(f"Best score (Brier Score) for {model_name}: {best_score:.4f} (lower is better)")
    print(f"Best parameters: {best_params}")

    # --- Save artifacts locally ---

    # 1. Save the trained model using joblib.
    model_path = model_output_dir / "model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved trained model to {model_path}")

    # 2. Save an example of the input data.
    input_example = X.head()
    example_path = model_output_dir / "input_example.csv"
    input_example.to_csv(example_path, index=False)
    print(f"Saved input example to {example_path}")
    
    # 3. Save metadata including parameters, score, and features.
    metadata = {
        "model_name": model_name,
        "brier_score": best_score,
        "best_parameters": best_params,
        "features": features
    }
    metadata_path = model_output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {metadata_path}")

    print(f"--- Finished training for: {model_name} ---\n")


def main():
    """
    Main function to orchestrate the loading of data and training of all models.
    """
    # Project paths are defined.
    try:
        project_root = Path(__file__).parent.parent.parent
    except NameError:
        project_root = Path('.').resolve().parent

    data_path = project_root / 'data' / 'preprocessed' / 'preprocessed_shots.csv'
    models_dir = project_root / 'models'
    
    # Create the main models directory if it doesn't exist.
    models_dir.mkdir(exist_ok=True)

    # The preprocessed data is loaded from the specified path.
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the preprocessed data exists.")
        return
    print("Data loaded successfully.")

    # The target variable and all available features are defined.
    all_cols = df.columns.tolist()
    target_variable = 'target'
    
    # The target variable is removed from the list of all columns to get all possible features.
    all_features = [col for col in all_cols if col != target_variable]

    # A base set of features is defined.
    base_features = [
        'X', 'Y', 'distance_to_goal', 'angle_to_goal', 
        'zone_central', 'zone_wide'
    ]
    
    # Features related to 'situation' and 'shotType' are identified.
    situation_features = [col for col in all_features if col.startswith('situation_')]
    shottype_features = [col for col in all_features if col.startswith('shotType_')]
    
    # Configurations for each model to be trained, specifying their feature sets, are defined.
    model_configs = {
        'basic_model': base_features,
        'situation_model': base_features + situation_features,
        'shottype_model': base_features + shottype_features,
        'advanced_model': all_features
    }

    # Each model configuration is iterated through to train and save the model.
    for name, features in model_configs.items():
        # Verification occurs that all required features for the model exist in the dataframe.
        missing_feats = [feat for feat in features if feat not in df.columns]
        if missing_feats:
            print(f"Warning: Missing columns for model '{name}': {missing_feats}. Skipping this model.")
            continue
        
        # Call the training and saving function.
        train_and_save_model(name, features, df, target_variable, models_dir)

    print("--- All models have been trained and saved. ---")
    print(f"Check the '{models_dir.name}' directory to see your saved models and their artifacts.")

if __name__ == '__main__':
    main()