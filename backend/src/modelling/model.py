import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import mlflow
import mlflow.sklearn

def train_and_log_model(model_name: str, features: list, data: pd.DataFrame, target_col: str, registered_model_name: str):
    """
    Trains a RandomForestClassifier, logs the experiment, and registers the model
    in the MLflow Model Registry, which handles versioning.

    Args:
        model_name (str): A descriptive name for the model run (e.g., 'basic_model').
        features (list): The list of feature column names to use for training.
        data (pd.DataFrame): The full dataframe containing features and the target.
        target_col (str): The name of the target variable column.
        registered_model_name (str): The name for the model in the Model Registry.
                                     MLflow will automatically version models under this name.
    """
    # Start a new MLflow run for this specific model training session.
    with mlflow.start_run(run_name=model_name):
        print(f"--- Starting training for: {model_name} ---")

        # Log a tag to easily identify the model type.
        mlflow.set_tag("model_name", model_name)
        mlflow.log_param("num_features", len(features))

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
        best_score = -bayes_search.best_score_
        best_params = bayes_search.best_params_

        print(f"Best score (Brier Score) for {model_name}: {best_score:.4f} (lower is better)")
        print(f"Best parameters: {best_params}")

        # Log the best hyperparameters and metric to the MLflow run.
        mlflow.log_params(best_params)
        mlflow.log_metric("brier_score", best_score)

        # The feature schema is logged as a dictionary artifact.
        schema = {'features': features}
        mlflow.log_dict(schema, "schema.json")
        print(f"Logged schema to MLflow artifact store.")

        # Log the model and register it, which enables automatic versioning.
        print(f"Registering model under the name: {registered_model_name}")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X.head(),
            registered_model_name=registered_model_name
        )
        print(f"Logged and Registered model successfully.")
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
    
    # Set an MLflow experiment to group all related runs.
    mlflow.set_experiment("Shot_Prediction_Models")

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

    # Each model configuration is iterated through to train and register the model.
    for name, features in model_configs.items():
        # Verification occurs that all required features for the model exist in the dataframe.
        missing_feats = [feat for feat in features if feat not in df.columns]
        if missing_feats:
            print(f"Warning: Missing columns for model '{name}': {missing_feats}. Skipping this model.")
            continue
        
        # Define a consistent name for the Model Registry.
        registry_name = f"{name}-xg"
        
        # Call the training function, passing the name for the registry.
        train_and_log_model(name, features, df, target_variable, registry_name)

    print("--- All models have been trained and registered in MLflow. ---")
    print("Run 'mlflow ui' and click the 'Models' tab to see your versioned models.")

if __name__ == '__main__':
    main()