from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import yaml
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sys
sys.path.append('src/models')
from ensemble_model import EnsembleModel

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

class NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def preprocess_input(data_dict):
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Define binary columns (0/1)
    binary_cols = ['Gender', 'Married', 'PhoneService', 'InternetService', 'PaperlessBilling']
    
    # Define categorical columns (Yes/No/Unknown)
    categorical_cols = [
        'Offer', 'MultipleLines', 'InternetType', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtectionPlan', 'PremiumTechSupport', 'StreamingTV',
        'StreamingMovies', 'StreamingMusic', 'UnlimitedData', 'Contract',
        'PaymentMethod'
    ]
    
    # Define numerical columns
    numerical_cols = [
        'Age', 'NumberofDependents', 'NumberofReferrals', 'TenureinMonths',
        'AvgMonthlyLongDistanceCharges', 'AvgMonthlyGBDownload', 'MonthlyCharge',
        'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges',
        'TotalLongDistanceCharges', 'TotalRevenue', 'Population'
    ]
    
    # Process binary columns
    for col in binary_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise').astype(int)
                if not df[col].isin([0, 1]).all():
                    raise ValueError(f"Invalid value for {col}. Must be 0 or 1")
            except Exception as e:
                raise ValueError(f"Invalid value for {col}. Must be 0 or 1")
    
    # Process categorical columns
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        if col == 'Offer':
            valid_values = ['Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E', 'None']
        elif col == 'InternetType':
            valid_values = ['Cable', 'DSL', 'Fiber Optic', 'Unknown']
        elif col == 'Contract':
            valid_values = ['Month-to-Month', 'One Year', 'Two Year']
        elif col == 'PaymentMethod':
            valid_values = ['Bank Withdrawal', 'Credit Card', 'Mailed Check']
        else:
            valid_values = ['Yes', 'No', 'Unknown']
            
        if not df[col].isin(valid_values).all():
            raise ValueError(f"Invalid value for {col}. Must be one of: {', '.join(valid_values)}")
    
    # Process numerical columns
    for col in numerical_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise').astype(float)
            except Exception as e:
                raise ValueError(f"Invalid value for {col}. Must be a number")
    
    return df

def predict(data):
    try:
        model_path = os.path.join("models", "model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = joblib.load(model_path)
        
        # Preprocess the input data
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)
        
        # Handle both single and multiple predictions
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]
        if isinstance(probability, np.ndarray):
            probability = probability[0][1] if probability.ndim > 1 else probability[0]
        
        # Format the response
        result = f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}\n"
        result += f"Churn Probability: {probability:.2%}"
        
        return result
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

def validate_input(dict_request):
    # Define binary columns that should be 0 or 1
    binary_cols = ['Gender', 'Married', 'PhoneService', 'InternetService', 'PaperlessBilling']
    
    # Define categorical columns that should be Yes/No/Unknown
    categorical_cols = [
        'Offer', 'MultipleLines', 'InternetType', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtectionPlan', 'PremiumTechSupport', 'StreamingTV',
        'StreamingMovies', 'StreamingMusic', 'UnlimitedData', 'Contract',
        'PaymentMethod'
    ]
    
    # Define numerical columns
    numerical_cols = [
        'Age', 'NumberofDependents', 'NumberofReferrals', 'TenureinMonths',
        'AvgMonthlyLongDistanceCharges', 'AvgMonthlyGBDownload', 'MonthlyCharge',
        'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges',
        'TotalLongDistanceCharges', 'TotalRevenue', 'Population'
    ]
    
    for key, val in dict_request.items():
        if key in binary_cols:
            try:
                val = int(val)
                if val not in [0, 1]:
                    raise ValueError(f"Invalid value for {key}. Must be 0 or 1")
            except ValueError:
                raise NotANumber(f"Invalid value for {key}. Must be 0 or 1")
        elif key in categorical_cols:
            if key == 'Offer':
                if val not in ['Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E', 'None']:
                    raise ValueError(f"Invalid value for {key}")
            elif key in ['InternetType', 'Contract', 'PaymentMethod']:
                # These have specific allowed values
                if key == 'InternetType' and val not in ['Cable', 'DSL', 'Fiber Optic', 'Unknown']:
                    raise ValueError(f"Invalid value for {key}")
                elif key == 'Contract' and val not in ['Month-to-Month', 'One Year', 'Two Year']:
                    raise ValueError(f"Invalid value for {key}")
                elif key == 'PaymentMethod' and val not in ['Bank Withdrawal', 'Credit Card', 'Mailed Check']:
                    raise ValueError(f"Invalid value for {key}")
            elif val not in ['Yes', 'No', 'Unknown']:
                raise ValueError(f"Invalid value for {key}. Must be 'Yes', 'No', or 'Unknown'")
        elif key in numerical_cols:
            try:
                val = float(val)
                # Add specific validation for Age
                if key == 'Age' and (val < 0 or val > 100):
                    raise ValueError(f"Age must be between 0 and 100 years")
            except ValueError as e:
                if "Age must be between" in str(e):
                    raise ValueError(str(e))
                raise NotANumber(f"Invalid value for {key}. Must be a number")
    return True

def form_response(dict_request):
    try:
        if validate_input(dict_request):
            response = predict(dict_request)
            return response
    except NotANumber as e:
        response = str(e)
        return response
    except ValueError as e:
        response = str(e)
        return response
    except Exception as e:
        response = f"Error: {str(e)}"
        return response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req)
                return render_template("index.html", response=response)
        except Exception as e:
            print(e)
            error = {"error": str(e)}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)