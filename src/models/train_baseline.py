# file: src/models/train_baseline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import os
import joblib

def train_baseline_model(ticker: str):
    """
    Trains a baseline RandomForestClassifier on technical indicators only.
    """
    # Load final dataset
    data_path = f"data/processed/{ticker}_final_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Final dataset not found: {data_path}")
    
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # Select features (technical indicators only)
    feature_columns = [
        'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
        'MACDs_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0',
        'BBP_20_2.0', 'close', 'high', 'low', 'open', 'volume'
    ]
    X = df[feature_columns]
    y = df['target']
    
    # Time-series split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training baseline model on {len(X_train)} samples...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nBaseline Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker}_baseline_model.joblib")
    joblib.dump(model, model_path)
    print(f"Baseline model saved to {model_path}")
    
    # Feature importances
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))

if __name__ == '__main__':
    TICKER = "AAPL"
    train_baseline_model(TICKER)