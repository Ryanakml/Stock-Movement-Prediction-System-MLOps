# file: src/models/train_lstm.py
import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(X, y, time_steps=1):
    """
    Creates sequences of data for LSTM model.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_lstm_model(ticker: str, time_steps: int = 30):
    """
    Trains an LSTM model with both technical and sentiment features.
    """
    print("Starting Training LSTM ...")
    print("[1/7] Loading dataset...")
    data_path = f"data/processed/{ticker}_final_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Final dataset not found: {data_path}")
    
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    print(f"   Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # All columns except target are features
    feature_columns = df.columns.drop('target')

    # Keep only numeric features
    X = df[feature_columns].select_dtypes(include=[np.number])
    y = df['target']

    print("Features used for training:", X.columns.tolist())
    print("Non-numeric columns dropped:", set(df[feature_columns].columns) - set(X.columns))

    y = df['target']
    
    print("[2/7] Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    print("   Features scaled.")

    print("[3/7] Creating sequences...")
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    print(f"   Sequences created: {X_seq.shape}")

    print("[4/7] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("[5/7] Building LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("   Model compiled.")

    print("[6/7] Training model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.1, # last 10% of training data
        callbacks=[early_stopping],
        shuffle=False,
        verbose=1  # shows live progress bar per epoch
    )

    print("[7/7] Evaluating model...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nLSTM Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    print("Saving model and scaler...")
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, f"{ticker}_lstm_model.h5"))
    joblib.dump(scaler, os.path.join(model_dir, f"{ticker}_scaler.joblib"))
    print(f"LSTM model and scaler saved to {model_dir}")

if __name__ == '__main__':
    TICKER = "AAPL"
    train_lstm_model(TICKER, time_steps=5)