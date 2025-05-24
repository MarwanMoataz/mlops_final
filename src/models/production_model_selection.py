import joblib
import mlflow
import argparse
import os
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient

def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    mlflow.set_tracking_uri(remote_server_uri)
    
    # Get the experiment ID dynamically
    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
    if experiment is None:
        raise Exception(f"Experiment '{mlflow_config['experiment_name']}' does not exist")
    
    # Search runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if len(runs) == 0:
        raise Exception("No runs found in the experiment")
    
    max_accuracy = max(runs["metrics.accuracy"])
    max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]
    
    # Load the model directly from the run
    model_uri = f"runs:/{max_accuracy_run_id}/model"
    
    try:
        # Register the model if it doesn't exist
        try:
            mlflow.register_model(model_uri, model_name)
        except Exception as e:
            print(f"Model {model_name} already exists. Proceeding with version transition.")
        
        client = MlflowClient()
        
        # Search for the specific version associated with this run
        for mv in client.search_model_versions(f"name='{model_name}'"):
            mv = dict(mv)
            if mv["run_id"] == max_accuracy_run_id:
                current_version = mv["version"]
                pprint(mv, indent=4)
                client.transition_model_version_stage(
                    name=model_name,
                    version=current_version,
                    stage="Production"
                )
                # Load and save the production model
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                joblib.dump(loaded_model, model_dir)
                print(f"Model version {current_version} has been promoted to production")
                return
            
        raise ValueError(f"No model version found with run_id {max_accuracy_run_id}")
        
    except Exception as e:
        raise Exception(f"Error in model registration process: {str(e)}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)