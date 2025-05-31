import joblib
import mlflow
import argparse
import os
import json
import yaml
from pprint import pprint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.ensemble_model import EnsembleModel

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def log_production_model(config_path):
    config = read_params(config_path)
    model_dir = config["model_dir"]
    trained_model_path = "models/trained_model.joblib"

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    
    try:
        # Load the model directly from the trained model file
        if not os.path.exists(trained_model_path):
            raise Exception(f"Trained model file not found at {trained_model_path}")
            
        print(f"Loading model from {trained_model_path}")
        loaded_model = joblib.load(trained_model_path)
        
        # Save the model to production
        joblib.dump(loaded_model, model_dir)
        print(f"Model saved to: {model_dir}")
        
        # Copy label encoder to production if it exists
        label_encoder_path = "models/label_encoder.joblib"
        if os.path.exists(label_encoder_path):
            production_label_encoder_path = os.path.join(os.path.dirname(model_dir), "label_encoder.joblib")
            joblib.dump(joblib.load(label_encoder_path), production_label_encoder_path)
            print("Label encoder copied to production")
        
        print("Model has been successfully loaded and saved to production")
        
    except Exception as e:
        raise Exception(f"Error in model loading process: {str(e)}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)