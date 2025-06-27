import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

def train_and_save_model(model_name: str, features: list, data: pd.DataFrame, target_col: str, output_root: Path):
    """
    Trains a RandomForestClassifier using BayesSearchCV, saves the best model,
    and its feature schema.

    Args:
        model_name (str): A descriptive name for the model (e.g., 'basic_model').
        features (list): The list of feature column names to use for training.
        data (pd.DataFrame): The full dataframe containing features and the target.
        target_col (str): The name of the target variable column.
        output_root (Path): The root directory to save the model folders to.
    """
    print(f"--- Starting training for: {model_name} ---")

    # Data is prepared for training.
    X = data[features]
    y = data[target_col]
    
    # The output directory for the current model is created if it does not exist.
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {model_output_dir}")

    # Hyperparameter search space.
    search_spaces = {
        'n_estimators': Integer(100, 500),
        'max_depth': Integer(10, 40),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2'])
    }

    # BayesSearchCV is used for hyperparameter optimization.
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
    print(f"Best score (Brier Score) for {model_name}: {-bayes_search.best_score_:.4f} (lower is better)")
    print(f"Best parameters: {bayes_search.best_params_}")

    # The trained model is saved to a joblib file.
    model_path = model_output_dir / 'model.joblib'
    joblib.dump(best_model, model_path)
    print(f"Saved model to: {model_path}")

    # The feature schema, which lists the features used for training this model, is saved.
    schema = {'features': features}
    schema_path = model_output_dir / 'schema.json'
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=4)
    print(f"Saved schema to: {schema_path}")
    print(f"--- Finished training for: {model_name} ---\n")


def main():
    """
    Main function to orchestrate the loading of data and training of all models.
    """
    # Project paths are defined
    try:
        project_root = Path(__file__).parent.parent.parent
    except NameError:
        project_root = Path('.').resolve().parent

    data_path = project_root / 'data' / 'preprocessed' / 'preprocessed_shots.csv'
    models_output_dir = project_root / 'models'

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
        train_and_save_model(name, features, df, target_variable, models_output_dir)

    print("--- All models have been trained and saved. ---")

if __name__ == '__main__':
    main()