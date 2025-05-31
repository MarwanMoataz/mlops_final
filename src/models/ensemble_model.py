import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE

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