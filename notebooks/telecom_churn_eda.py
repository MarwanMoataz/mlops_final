#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Telecom Customer Churn EDA
This script performs a comprehensive Exploratory Data Analysis (EDA) on the Telecom Customer Churn dataset
following MLOps best practices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import missingno as msno
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load and return the telecom churn dataset."""
    print("Loading dataset...")
    try:
        df = pd.read_csv('telecom_customer_churn.csv')
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: telecom_customer_churn.csv not found in the current directory.")
        print("Please make sure the file exists in the notebooks directory.")
        return None

def initial_inspection(df):
    """Perform initial data inspection."""
    if df is None:
        return
        
    print("\n=== Initial Data Inspection ===")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nColumn Information:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe(include='all'))

def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    if df is None:
        return
        
    print("\n=== Missing Value Analysis ===")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print("\nMissing Value Statistics:")
        print(missing_df)
    else:
        print("\nNo missing values found in the dataset.")
    
    # Create missing value visualization
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title('Missing Values Matrix')
    plt.savefig('missing_values_matrix.png')
    plt.close()

def analyze_categorical_features(df):
    """Analyze categorical features in the dataset."""
    if df is None:
        return
        
    print("\n=== Categorical Feature Analysis ===")
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        print(f"\nUnique values in {col}:")
        print(df[col].value_counts())
        
        # Create bar plot for categorical features
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'categorical_{col}_distribution.png')
        plt.close()

def analyze_numerical_features(df):
    """Analyze numerical features in the dataset."""
    if df is None:
        return
        
    print("\n=== Numerical Feature Analysis ===")
    
    # Print all column names to help identify the target column
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Find the target column (it might be named differently)
    target_column = None
    possible_target_names = ['churn', 'Churn', 'CHURN', 'target', 'Target', 'TARGET']
    for col in possible_target_names:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        print("\nWarning: Could not find target column. Using all numerical columns without target grouping.")
        target_column = None
    
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if target_column in numerical_columns:
        numerical_columns = numerical_columns.drop(target_column)
    
    # Create distribution plots for numerical features
    n_cols = 3
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_columns):
        if target_column:
            sns.histplot(data=df, x=col, hue=target_column, ax=axes[idx])
        else:
            sns.histplot(data=df, x=col, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
    
    # Remove empty subplots
    for idx in range(len(numerical_columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('numerical_features_distribution.png')
    plt.close()

def analyze_correlations(df):
    """Analyze correlations between features."""
    if df is None:
        return
        
    print("\n=== Correlation Analysis ===")
    
    # Find the target column
    target_column = None
    possible_target_names = ['churn', 'Churn', 'CHURN', 'target', 'Target', 'TARGET']
    for col in possible_target_names:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        print("\nWarning: Could not find target column for correlation analysis.")
        return
    
    correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Plot correlation with target variable
    target_correlations = correlation_matrix[target_column].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=target_correlations.values, y=target_correlations.index)
    plt.title(f'Feature Correlations with {target_column}')
    plt.tight_layout()
    plt.savefig('target_correlations.png')
    plt.close()

def analyze_feature_importance(df):
    """Analyze feature importance using Random Forest."""
    if df is None:
        return
        
    print("\n=== Feature Importance Analysis ===")
    
    # Find the target column
    target_column = None
    possible_target_names = ['churn', 'Churn', 'CHURN', 'target', 'Target', 'TARGET']
    for col in possible_target_names:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        print("\nWarning: Could not find target column for feature importance analysis.")
        return
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != target_column:  # Don't encode the target column
            X[col] = le.fit_transform(X[col])
    
    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def check_data_quality(df):
    """Perform data quality checks."""
    if df is None:
        return
        
    print("\n=== Data Quality Checks ===")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    # Check for outliers using IQR method
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        return len(outliers)
    
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    outlier_summary = pd.DataFrame({
        'Column': numerical_columns,
        'Outliers': [detect_outliers(df, col) for col in numerical_columns]
    })
    
    print("\nOutlier Summary:")
    print(outlier_summary)
    
    # Find the target column
    target_column = None
    possible_target_names = ['churn', 'Churn', 'CHURN', 'target', 'Target', 'TARGET']
    for col in possible_target_names:
        if col in df.columns:
            target_column = col
            break
    
    if target_column:
        # Check for class imbalance
        class_balance = df[target_column].value_counts(normalize=True) * 100
        print(f"\nClass Balance for {target_column}:")
        print(class_balance)
    else:
        print("\nWarning: Could not find target column for class balance analysis.")

def main():
    """Main function to run the EDA."""
    # Create output directory for plots
    os.makedirs('eda_plots', exist_ok=True)
    os.chdir('eda_plots')
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Perform EDA
        initial_inspection(df)
        analyze_missing_values(df)
        analyze_categorical_features(df)
        analyze_numerical_features(df)
        analyze_correlations(df)
        analyze_feature_importance(df)
        check_data_quality(df)
        
        print("\nEDA completed successfully! All plots have been saved in the 'eda_plots' directory.")
    else:
        print("\nEDA could not be completed due to missing data file.")

if __name__ == "__main__":
    main() 