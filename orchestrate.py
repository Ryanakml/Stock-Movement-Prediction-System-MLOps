# file: orchestrate.py
from prefect import task, flow
import os
import subprocess

# --- Define Tasks ---
@task(retries=3, retry_delay_seconds=60)
def run_data_ingestion():
    """Task to run all data ingestion scripts."""
    print("Running data ingestion...")
    subprocess.run(["python", "src/data/price_ingestion.py"], check=True)
    subprocess.run(["python", "src/data/news_ingestion.py"], check=True)
    # subprocess.run(["python", "src/data/reddit_ingestion.py"], check=True) # Uncomment if using
    print("Data ingestion complete.")
    return True

@task
def run_feature_engineering(upstream_result: bool):
    """Task to run all feature engineering scripts."""
    if not upstream_result: return
    print("Running feature engineering...")
    subprocess.run(["python", "src/features/sentiment_analysis.py"], check=True)
    subprocess.run(["python", "src/features/technical_indicators.py"], check=True)
    subprocess.run(["python", "src/data/build_features.py"], check=True)
    print("Feature engineering complete.")
    return True

@task
def run_drift_detection(upstream_result: bool):
    """Task to run the drift detection script."""
    if not upstream_result: return False
    print("Running drift detection...")
    # This is a simplified call. The script needs to be adapted to return a status code or file.
    # For now, we assume it writes a result file.
    result = subprocess.run(["python", "src/monitoring/detect_drift.py"])
    # A real implementation would parse the output of the script to determine drift
    # For this demo, we'll simulate drift being detected.
    drift_detected = True # Simulate drift for demonstration
    print(f"Drift detection check complete. Drift detected: {drift_detected}")
    return drift_detected

@task
def run_model_retraining(drift_detected: bool):
    """Task to retrain the LSTM model if drift is detected."""
    if not drift_detected:
        print("No drift detected. Skipping retraining.")
        return None
    print("Drift detected. Starting model retraining...")
    subprocess.run(["python", "src/models/train_lstm.py"], check=True)
    print("Model retraining complete.")
    return "retrained"

@task
def trigger_deployment(retrain_status: str):
    """Task to trigger a new deployment on Render."""
    if retrain_status!= "retrained":
        print("Model not retrained. Skipping deployment.")
        return
    
    deploy_hook_url = os.getenv("RENDER_DEPLOY_HOOK_URL")
    if not deploy_hook_url:
        print("RENDER_DEPLOY_HOOK_URL not set. Cannot trigger deployment.")
        return
        
    print("Triggering deployment on Render...")
    response = subprocess.run(, capture_output=True)
    if response.returncode == 0:
        print("Deployment triggered successfully.")
    else:
        print(f"Failed to trigger deployment: {response.stderr.decode()}")

# --- Define Flow ---
@flow(name="Stock Prediction Retraining Flow")
def retraining_flow():
    ingestion_complete = run_data_ingestion()
    features_complete = run_feature_engineering(ingestion_complete)
    drift_detected = run_drift_detection(features_complete)
    retrain_status = run_model_retraining(drift_detected)
    trigger_deployment(retrain_status)

if __name__ == "__main__":
    # To run the flow and schedule it, you would use the Prefect CLI:
    # 1. `prefect server start` (in a separate terminal)
    # 2. `python orchestrate.py` (to register the flow)
    # 3. `prefect deployment build./orchestrate.py:retraining_flow -n stock-retraining -q default`
    # 4. `prefect deployment apply retraining_flow-deployment.yaml`
    # 5. Go to the Prefect UI to run or schedule the deployment.
    retraining_flow()