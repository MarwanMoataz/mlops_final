import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def accuracymeasures(y_test,predictions,avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ['0','1']
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe. Available columns: {df.columns.tolist()}")
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def create_preprocessing_pipeline(df):
    """
    Create preprocessing pipeline for categorical and numerical features
    """
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing steps
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ])
    
    return preprocessor, categorical_cols, numerical_cols

def save_confusion_matrix_plot(y_true, y_pred, model_name):
    """Save confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'plots/confusion_matrix_{model_name}.png')
    plt.close()

def get_pipeline(X, model, config):
    """Create the full pipeline with preprocessing, SMOTE, and model"""
    numeric_pipeline = SimpleImputer(strategy='mean')
    categorical_pipeline = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, config['data_processing']['numerical_features']),
            ('categorical', categorical_pipeline, config['data_processing']['categorical_features']),
        ], remainder='passthrough'
    )

    bundled_pipeline = imbpipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=config['raw_data_config']['random_state'])),
        ('scaler', MinMaxScaler()),
        ('model', model)
    ])
    
    return bundled_pipeline

class EnsembleModel:
    def __init__(self, models, config):
        self.models = models
        self.config = config
        self.pipelines = []
        
    def fit(self, X, y):
        for name, model in self.models:
            pipeline = get_pipeline(X, model, self.config)
            pipeline.fit(X, y)
            self.pipelines.append(pipeline)
        return self
    
    def predict(self, X):
        predictions = []
        for pipeline in self.pipelines:
            pred_proba = pipeline.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        ensemble_pred_proba = np.mean(predictions, axis=0)
        return (ensemble_pred_proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        predictions = []
        for pipeline in self.pipelines:
            pred_proba = pipeline.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        return np.mean(predictions, axis=0)

def train_and_evaluate(config_path):
    config = read_params(config_path)
    
    # Create necessary directories
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Load the cleaned data
    df = pd.read_csv(config['processed_data_config']['cleaned_data_csv'])
    
    # Prepare features and target
    X = df.drop(config['data_processing']['columns_to_remove'], axis=1)
    y = df[config['raw_data_config']['target']]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['raw_data_config']['train_test_split_ratio'],
        random_state=config['raw_data_config']['random_state'],
        stratify=y
    )
    
    # Save train/test splits
    X_train.to_csv(config['processed_data_config']['train_data_csv'], index=False)
    X_test.to_csv(config['processed_data_config']['test_data_csv'], index=False)
    
    # Initialize MLflow
    mlflow.set_tracking_uri(config['mlflow_config']['remote_server_uri'])
    mlflow.set_experiment(config['mlflow_config']['experiment_name'])
    
    # Dictionary to store model results
    model_results = {}
    
    # Define all models
    models = {
        "XGBoost": XGBClassifier(**config['xgboost']),
        "LGBM": LGBMClassifier(),
        "RandomForest": RandomForestClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "ExtraTree": ExtraTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "KNeighbors": KNeighborsClassifier(),
        "Ridge": RidgeClassifier(),
        "SGD": SGDClassifier(),
        "Bagging": BaggingClassifier(),
        "BernoulliNB": BernoulliNB(),
        "SVC": SVC(probability=True),  # Enable probability estimates
        "CatBoost": CatBoostClassifier(silent=True)
    }
    
    # Train and evaluate individual models
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            logger.info(f"Training {model_name}")
            
            # Create and fit pipeline
            pipeline = get_pipeline(X_train, model, config)
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate ROC AUC if model supports predict_proba
            try:
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except (AttributeError, NotImplementedError):
                roc_auc = None
            
            # Log metrics
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            if roc_auc is not None:
                metrics["roc_auc"] = roc_auc
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix and classification report
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            
            # Save metrics to files
            with open(f"metrics/confusion_matrix_{model_name}.txt", "w") as f:
                f.write(str(cm))
            with open(f"metrics/classification_report_{model_name}.txt", "w") as f:
                f.write(cr)
            
            # Save confusion matrix plot
            save_confusion_matrix_plot(y_test, y_pred, model_name)
            
            # Log the model
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=f"{config['mlflow_config']['registered_model_name']}_{model_name}"
            )
            
            model_results[model_name] = {
                "model": pipeline,
                "accuracy": accuracy,
                "run_id": run.info.run_id
            }
    
    # Train and evaluate ensemble models
    ensemble_models = {
        "Ensemble_XGB_CatBoost_Bagging": [
            ('xgb', XGBClassifier(**config['xgboost'])),
            ('catboost', CatBoostClassifier(silent=True)),
            ('bagging', BaggingClassifier())
        ],
        "Ensemble_XGB_LGBM_CatBoost": [
            ('xgb', XGBClassifier(**config['xgboost'])),
            ('lgbm', LGBMClassifier()),
            ('catboost', CatBoostClassifier(silent=True))
        ],
        "Ensemble_XGB_RF_DT": [
            ('xgb', XGBClassifier(**config['xgboost'])),
            ('rf', RandomForestClassifier()),
            ('dt', DecisionTreeClassifier())
        ],
        "Ensemble_XGB_AdaBoost": [
            ('xgb', XGBClassifier(**config['xgboost'])),
            ('adaboost', AdaBoostClassifier())
        ]
    }
    
    for ensemble_name, estimators in ensemble_models.items():
        with mlflow.start_run(run_name=ensemble_name) as run:
            logger.info(f"Training {ensemble_name}")
            
            # Create and fit ensemble model
            ensemble = EnsembleModel(estimators, config)
            ensemble.fit(X_train, y_train)
            
            # Make predictions
            y_pred = ensemble.predict(X_test)
            y_pred_proba = ensemble.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            })
            
            # Log confusion matrix and classification report
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            
            # Save metrics to files
            with open(f"metrics/confusion_matrix_{ensemble_name}.txt", "w") as f:
                f.write(str(cm))
            with open(f"metrics/classification_report_{ensemble_name}.txt", "w") as f:
                f.write(cr)
            
            # Save confusion matrix plot
            save_confusion_matrix_plot(y_test, y_pred, ensemble_name)
            
            # Save ensemble predictions
            ensemble_results = pd.DataFrame({
                'true_label': y_test,
                'predicted_probability': y_pred_proba,
                'predicted_label': y_pred
            })
            ensemble_results.to_csv(f"models/{ensemble_name}_predictions.csv", index=False)
            
            model_results[ensemble_name] = {
                "model": ensemble,
                "accuracy": accuracy,
                "run_id": run.info.run_id
            }
    
    # Select best model based on accuracy
    best_model_name = max(model_results.items(), key=lambda x: x[1]["accuracy"])[0]
    best_model = model_results[best_model_name]["model"]
    best_run_id = model_results[best_model_name]["run_id"]
    
    # Save the best model info
    with open("best_model_info.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "run_id": best_run_id,
            "registered_model_name": f"{config['mlflow_config']['registered_model_name']}_{best_model_name}"
        }, f)
    
    # Save the best model locally
    joblib.dump(best_model, "models/trained_model.joblib")
    joblib.dump(best_model, config['model_webapp_dir'])
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best model saved to models/trained_model.joblib")
    logger.info(f"Model metrics logged to MLflow")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)



