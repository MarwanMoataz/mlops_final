import yaml
import argparse
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def model_monitoring(config_path):
    config = read_params(config_path)

    train_data_path = config["raw_data_config"]["raw_data_csv"]
    new_train_data_path = config["raw_data_config"]["new_train_data_csv"]
    target = config["raw_data_config"]["target"]
    monitor_dashboard_path = config["model_monitor"]["monitor_dashboard_html"]
    monitor_target = config["model_monitor"]["target_col_name"]

    ref = pd.read_csv(train_data_path)
    cur = pd.read_csv(new_train_data_path)

    # Rename target column to monitor_target (if needed)
    ref = ref.rename(columns={target: monitor_target}, inplace=False)
    cur = cur.rename(columns={target: monitor_target}, inplace=False)

    # Create column mapping with features info
    column_mapping = ColumnMapping()
    column_mapping.target = monitor_target
    # This assumes your config has "model_var" listing numerical features
    column_mapping.numerical_features = config["raw_data_config"]["model_var"]
    # Optionally, if you have categorical features:
    # column_mapping.categorical_features = ['state', 'international_plan', 'voice_mail_plan']

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])

    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)
    report.save_html(monitor_dashboard_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    model_monitoring(config_path=parsed_args.config)
