import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from scipy.stats import uniform, loguniform
import joblib
import json

def train_and_save_model(model_name: str, features: list, data: pd.DataFrame, target_col: str, models_dir: Path):
    """
    Trains a LogisticRegression model, and saves the model, an input example,
    and metadata to a dedicated local directory. Uses StratifiedKFold for cross-validation
    and performs a train-test split.

    Args:
        model_name (str): A descriptive name for the model (e.g., 'logistic_basic_model'),
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

    # Prepare data for training.
    X = data[features]
    y = data[target_col]

    # Split data into training and testing sets, ensuring stratification.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=104, stratify=y)
    
    # Create a pipeline to scale data and then apply logistic regression.
    # Using class_weight='balanced' to handle class imbalance directly in the model.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=104, max_iter=1000)) 
    ])

    # Hyperparameter search space for Logistic Regression using distributions for RandomizedSearchCV.
    search_spaces = {
        'logreg__penalty': ['l1', 'l2'],
        'logreg__C': loguniform(1e-4, 1e+4), # Use loguniform for C
        'logreg__solver': ['liblinear', 'saga']
    }

    # Use StratifiedKFold for cross-validation to maintain class proportions.
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=104)

    # RandomizedSearchCV for hyperparameter optimization with stratified cross-validation.
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_spaces,
        n_iter=50,
        cv=cv_strategy,
        scoring='neg_brier_score',
        n_jobs=-1,
        random_state=104,
        verbose=1,
        refit=True
    )

    print(f"Running RandomizedSearchCV for {model_name}...")
    random_search.fit(X_train, y_train) # Fit on training data

    best_model = random_search.best_estimator_
    best_score = -random_search.best_score_ # Convert neg_brier_score back to brier_score
    
    best_params = random_search.best_params_
    
    print(f"Best score (Brier Score) for {model_name}: {best_score:.4f} (lower is better)")
    print(f"Best parameters: {best_params}")

    # --- Save artifacts locally ---

    # Save the trained model pipeline.
    model_path = model_output_dir / "model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved trained model to {model_path}")

    # Save an example of the input data.
    input_example = X_train.head()
    example_path = model_output_dir / "input_example.csv"
    input_example.to_csv(example_path, index=False)
    print(f"Saved input example to {example_path}")
    
    # Save metadata.
    metadata = {
        "model_name": model_name,
        "model_type": "Logistic Regression",
        "brier_score": best_score,
        "best_parameters": best_params,
        "features": features,
        "test_set_size": len(X_test)
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
    # Define project paths.
    try:
        project_root = Path(__file__).parent.parent.parent
    except NameError:
        project_root = Path('.').resolve().parent

    data_path = project_root / 'data' / 'preprocessed' / 'preprocessed_shots.csv'
    models_dir = project_root / 'models'
    
    models_dir.mkdir(exist_ok=True)

    # Load preprocessed data.
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the preprocessed data exists.")
        return
    print("Data loaded successfully.")

    # Define target and features.
    all_cols = df.columns.tolist()
    target_variable = 'target'
    
    all_features = [col for col in all_cols if col != target_variable]

    base_features = [
        'X', 'Y', 'distance_to_goal', 'angle_to_goal', 
        'zone_central', 'zone_wide'
    ]
    
    situation_features = [col for col in all_features if col.startswith('situation_')]
    shottype_features = [col for col in all_features if col.startswith('shotType_')]
    
    # Configure models to be trained.
    model_configs = {
        'basic_model': base_features,
        'situation_model': base_features + situation_features,
        'shottype_model': base_features + shottype_features,
        'advanced_model': all_features
    }

    # Iterate through model configurations to train and save.
    for name, features in model_configs.items():
        # Verify all required features exist.
        missing_feats = [feat for feat in features if feat not in df.columns]
        if missing_feats:
            print(f"Warning: Missing columns for model '{name}': {missing_feats}. Skipping this model.")
            continue
        
        train_and_save_model(name, features, df, target_variable, models_dir)

    print("--- All models have been trained and saved. ---")
    print(f"Check the '{models_dir.name}' directory to see your saved models and their artifacts.")

if __name__ == '__main__':
    main()