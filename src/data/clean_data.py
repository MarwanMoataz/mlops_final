import pandas as pd
import numpy as np
from scipy import stats
import yaml
import os
from pathlib import Path
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def load_config():
    """Load configuration from params.yaml"""
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def clean_data():
    """Main function to clean and preprocess the data"""
    # Load configuration
    config = load_config()
    
    # Create output directories if they don't exist
    Path(config['processed_data_config']['train_data_csv']).parent.mkdir(parents=True, exist_ok=True)
    
    # Read the data
    df = pd.read_csv(config['external_data_config']['external_data_csv'])
    
    # Binary encoding
    df['Churn'] = df['Churn'].replace('Joined', 'Stayed')
    df['Churn'] = df['Churn'].replace(['Stayed','Churned'], [0,1])
    df['Married'] = df['Married'].replace(['Yes','No'], [1,0])
    df['Gender'] = df['Gender'].replace(['Female','Male'], [0,1])  # 1 = male, 0 = female
    df['PaperlessBilling'] = df['PaperlessBilling'].replace(['Yes','No'], [1,0])
    df['InternetService'] = df['InternetService'].replace(['Yes','No'], [1,0])
    df['PhoneService'] = df['PhoneService'].replace(['Yes','No'], [1,0])  # 1 = Yes, 0 = No
    
    # Fill missing values
    df['AvgMonthlyGBDownload'] = df['AvgMonthlyGBDownload'].fillna(0)
    df['AvgMonthlyLongDistanceCharges'] = df['AvgMonthlyLongDistanceCharges'].fillna(0)
    
    # Fill text columns with 'Unknown'
    cols_to_change = [
        'ChurnCategory', 'ChurnReason', 'UnlimitedData', 'StreamingMusic',
        'StreamingMovies', 'StreamingTV', 'PremiumTechSupport', 'DeviceProtectionPlan',
        'OnlineSecurity', 'OnlineBackup', 'InternetType', 'MultipleLines'
    ]
    df[cols_to_change] = df[cols_to_change].fillna('Unknown')
    
    # Remove outliers
    df = df[(np.abs(stats.zscore(df['TotalRevenue'])) < 3)]
    df = df[(np.abs(stats.zscore(df['Population'])) < 3)]
    
    # Save processed data
    df.to_csv(config['processed_data_config']['cleaned_data_csv'], index=False)
    
    print(f"Data cleaned and saved to {config['processed_data_config']['cleaned_data_csv']}")

if __name__ == "__main__":
    clean_data() 