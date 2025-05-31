# Customer Churn Prediction MLOps Project

This project implements an end-to-end MLOps pipeline for predicting customer churn using machine learning. The system includes data processing, model training, evaluation, and a web interface for making predictions.

## Project Structure

```
.
├── data/                  # Data directory
│   ├── external/         # External data sources
│   ├── raw/             # Raw data files
│   └── processed/       # Processed data files
├── models/              # Trained models
├── notebooks/          # Jupyter notebooks for analysis
├── reports/           # Generated reports and visualizations
├── src/               # Source code
│   └── models/       # Model training and evaluation code
│ 
├── webapp/           # Web application
│   ├── static/      # Static files
│   ├── templates/   # HTML templates
│   └── model_webapp_dir/  # Model files for web app
├── artifacts/        # MLflow artifacts
├── metrics/          # Model metrics
├── plots/           # Generated plots
└── params.yaml      # Configuration parameters
```

## Prerequisites

- Python 3.8 or higher
- Git
- DVC (Data Version Control)
- MLflow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MarwanMoataz/mlops_final
cd mlops_final
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install additional required packages:
```bash
pip install mlflow dvc scikit-learn pandas numpy xgboost lightgbm catboost flask
```

## Setup MLflow

1. Start the MLflow server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 -p 1234
```

2. Keep this terminal running while working with the project.

## Data Pipeline

The project uses DVC for data versioning and pipeline management. The pipeline consists of the following stages:

1. **Data Cleaning** (`clean_data`):
   - Processes raw data
   - Handles missing values
   - Performs feature engineering
   - Outputs cleaned data to `data/processed/CleanedDF.csv`

2. **Model Training** (`train_model`):
   - Trains multiple models (XGBoost, LightGBM, CatBoost)
   - Performs hyperparameter tuning
   - Generates model metrics and plots
   - Saves the best model

3. **Production Model Selection** (`log_production_model`):
   - Selects the best performing model
   - Logs the model to MLflow
   - Saves the model for production use

## Running the Pipeline

1. Initialize DVC (if not already done):
```bash
dvc init
```

2. Run the complete pipeline:
```bash
dvc repro
```

3. To run specific stages:
```bash
dvc repro clean_data
dvc repro train_model
dvc repro log_production_model
```

## Web Application

The project includes a Flask web application for making predictions:

1. Start the web application:
```bash
python app.py
```

2. Access the application at `http://127.0.0.1:5000`

The web interface allows users to:
- Input customer data
- Get churn predictions
- View prediction probabilities

## Model Features

The model uses the following features:

### Numerical Features
- Age
- Number of Dependents
- Population
- Number of Referrals
- Tenure in Months
- Average Monthly Long Distance Charges
- Average Monthly GB Download
- Monthly Charge
- Total Charges
- Total Refunds
- Total Extra Data Charges
- Total Long Distance Charges
- Total Revenue

### Categorical Features
- Gender
- Offer
- Married
- Phone Service
- Multiple Lines
- Internet Service
- Internet Type
- Online Security
- Online Backup
- Device Protection Plan
- Premium Tech Support
- Streaming TV
- Streaming Movies
- Streaming Music
- Unlimited Data
- Contract
- Paperless Billing
- Payment Method

## Model Performance

The model's performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

Metrics are saved in the `metrics` directory and can be viewed in MLflow.

## Monitoring

The project includes model monitoring capabilities:
- Data drift detection
- Target drift detection
- Performance monitoring

To run the model monitoring:

1. Ensure you have new data in the correct location:
```bash
data/raw/newdata.csv
```

2. Run the model monitoring script:
```bash
python src/models/model_monitor.py
```

This will:
- Compare the new data with the training data
- Generate data drift reports
- Create monitoring dashboards in the `reports` directory

The monitoring dashboard will show:
- Feature drift analysis
- Target distribution changes
- Model performance metrics
- Data quality metrics

Monitoring dashboards are generated in the `reports` directory and can be viewed in a web browser.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request



