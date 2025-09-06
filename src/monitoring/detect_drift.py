# file: src/monitoring/detect_drift.py
import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json

def check_data_drift(reference_path: str, current_path: str, report_path: str) -> bool:
    """
    Compares two datasets for data drift and returns if drift is detected.
    """
    ref_data = pd.read_csv(reference_path)
    curr_data = pd.read_csv(current_path)
    
    # Evidently needs column mapping if target/prediction are present
    # For data drift, we can often omit this if we only check input features
    
    data_drift_report = Report(metrics=)
    data_drift_report.run(reference_data=ref_data, current_data=curr_data)
    
    # Save the interactive HTML report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    data_drift_report.save_html(report_path)
    
    # Programmatically check for drift
    report_dict = data_drift_report.as_dict()
    is_drifted = report_dict['metrics']['result']['dataset_drift']
    
    print(f"Drift detected: {is_drifted}")
    print(f"Drift report saved to: {report_path}")
    
    return is_drifted

if __name__ == '__main__':
    TICKER = "AAPL"
    # In a real scenario, current_data would be recently collected live data
    # Here we simulate it by splitting our dataset
    df = pd.read_csv(f"data/processed/{TICKER}_final_dataset.csv")
    
    # Reference data is the first half, current data is the second half
    ref_df = df.iloc[:len(df)//2]
    curr_df = df.iloc[len(df)//2:]
    
    ref_path = "data/monitoring/reference_data.csv"
    curr_path = "data/monitoring/current_data.csv"
    ref_df.to_csv(ref_path, index=False)
    curr_df.to_csv(curr_path, index=False)
    
    check_data_drift(
        reference_path=ref_path,
        current_path=curr_path,
        report_path="reports/data_drift_report.html"
    )