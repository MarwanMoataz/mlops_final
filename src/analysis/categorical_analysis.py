import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data(data_path):
    """Load the dataset"""
    return pd.read_csv(data_path)

def perform_chi_square_test(df, categorical_cols, target_col='churn'):
    """
    Perform Chi-Square test for each categorical column against the target
    """
    results = []
    for col in categorical_cols:
        # Create contingency table
        contingency = pd.crosstab(df[col], df[target_col])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Calculate Cramer's V (measure of association)
        n = contingency.sum().sum()
        cramer_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        results.append({
            'Feature': col,
            'Chi-Square': chi2,
            'p-value': p_value,
            "Cramer's V": cramer_v
        })
    
    return pd.DataFrame(results)

def plot_categorical_features(df, categorical_cols, target_col='churn'):
    """
    Create visualizations for categorical features
    """
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        
        # Calculate percentages
        prop_df = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        
        # Plot
        ax = prop_df.plot(kind='bar', stacked=True)
        plt.title(f'Churn Rate by {col}')
        plt.xlabel(col)
        plt.ylabel('Percentage')
        plt.legend(title='Churn', labels=['No', 'Yes'])
        
        # Add percentage labels
        for c in ax.containers:
            labels = [f'{v:.1f}%' if v > 0 else '' for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type='center')
        
        plt.tight_layout()
        plt.savefig(f'reports/figures/{col}_vs_churn.png')
        plt.close()

def analyze_categorical_features(data_path):
    """
    Main function to analyze categorical features
    """
    # Load data
    df = load_data(data_path)
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'churn' in categorical_cols:
        categorical_cols.remove('churn')
    
    print("Categorical features found:", categorical_cols)
    
    # Perform Chi-Square test
    chi_square_results = perform_chi_square_test(df, categorical_cols)
    print("\nChi-Square Test Results:")
    print(chi_square_results.sort_values('p-value'))
    
    # Save results to CSV
    chi_square_results.to_csv('reports/categorical_analysis/chi_square_results.csv', index=False)
    
    # Create visualizations
    plot_categorical_features(df, categorical_cols)
    
    # Print interpretation
    print("\nInterpretation:")
    print("1. p-value < 0.05 indicates a significant relationship with churn")
    print("2. Cramer's V interpretation:")
    print("   - 0.1 to 0.3: Weak association")
    print("   - 0.3 to 0.5: Moderate association")
    print("   - > 0.5: Strong association")
    
    # Return the most influential features
    significant_features = chi_square_results[chi_square_results['p-value'] < 0.05].sort_values("Cramer's V", ascending=False)
    print("\nMost influential categorical features:")
    print(significant_features[['Feature', "Cramer's V"]].head())

if __name__ == "__main__":
    data_path = "data/processed/churn_train.csv"
    analyze_categorical_features(data_path) 