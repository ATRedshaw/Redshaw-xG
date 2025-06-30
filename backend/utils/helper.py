import os
import yaml
from typing import Dict, List

def get_model_folders() -> List[str]:
    """Get all model folders in the mlruns/models directory.

    Returns:
        List[str]: List of model folder names.
    """
    model_dir = os.path.join('mlruns', 'models')
    return [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]

def get_model_versions(model_name: str) -> List[str]:
    """Get all versions of a model in the mlruns/models directory.

    Args:
        model_name (str): Name of the model.

    Returns:
        List[str]: List of model version names.
    """
    model_dir = os.path.join('mlruns', 'models', model_name)
    return [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]

def get_prod_models(models: Dict[str, List[str]]) -> Dict[str, str]:
    """Get the production models from the app_config.yaml file.

    Args:
        models (Dict[str, List[str]]): A dictionary mapping model names to their versions.

    Returns:
        Dict[str, str]: A dictionary mapping model names to their storage locations.
    """
    with open('app_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    prod_models = {}

    for model, versions in models.items():
        model_key = model + '-xg'
        version_info = config.get(model_key, {'version': 'latest'})
        specified_version = version_info['version']
        
        # Determine the version to use
        if specified_version == 'latest' or specified_version not in versions:
            latest_version = sorted(versions, reverse=True)[0]
            version_to_use = latest_version
        else:
            version_to_use = specified_version

        meta_yaml_path = os.path.join('mlruns', 'models', model, version_to_use, 'meta.yaml')
        with open(meta_yaml_path, 'r') as f:
            meta_config = yaml.safe_load(f)
            storage_location = meta_config['storage_location']
            # Extract relevant storage path
            prod_models[model] = storage_location.split('backend/', 1)[-1]

    return prod_models

def return_model_paths():
    model_folders = get_model_folders()
    models = {model_folder: get_model_versions(model_folder) for model_folder in model_folders}
    prod_models = get_prod_models(models)
    return prod_models

if __name__ == "__main__":
    print(return_model_paths())