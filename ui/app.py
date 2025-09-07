# file: ui/app.py
import streamlit as st
import requests
import pandas as pd
import joblib
import os
os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/.streamlit'
os.environ['STREAMLIT_CREDENT statIALS_DIR'] = '/tmp/.streamlit'

# Utility function to get the latest 30 days of features
def get_latest_features():
    st.warning("Using pre-saved sample data for demonstration purposes.")
    # Fix: Explicitly use 'Date' as the index column
    sample_data = pd.read_csv("AAPL_final_dataset.csv", index_col='Date')
    
    # Drop the 'Unnamed: 0' column which is causing the error
    if 'Unnamed: 0' in sample_data.columns:
        sample_data = sample_data.drop('Unnamed: 0', axis=1)
    
    # Convert the index to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(sample_data.index):
        try:
            sample_data.index = pd.to_datetime(sample_data.index)
        except:
            pass  # If conversion fails, keep the original index
    
    scaler = joblib.load("models/AAPL_scaler.joblib")
    feature_columns = sample_data.columns.drop('target')
    scaled_features = scaler.transform(sample_data[feature_columns])
    
    latest_sequence = scaled_features[-30:]
    return latest_sequence.tolist()


# --- Streamlit App ---
st.title("Stock Price Movement Prediction")

# Use Render API URL with the correct endpoint
API_URL = "https://stock-movement-prediction-system-mlops.onrender.com/predict"

st.write("This dashboard predicts the next day's stock price movement for AAPL (Apple Inc.).")
st.write("Prediction: 1 for UP, 0 for DOWN.")

if st.button("Get Latest Prediction"):
    with st.spinner("Fetching latest data and making prediction..."):
        try:
            # Get the latest feature data
            features = get_latest_features()
            
            # Call the prediction API
            response = requests.post(API_URL, json={"data": features})
            response.raise_for_status()
            
            result = response.json()
            
            # Display the result
            st.success("Prediction received!")
            prediction = result.get("prediction")
            probability = result.get("probability_up")
            
            if prediction == 1:
                st.metric(label="Predicted Movement", value="UP", delta=f"{probability:.2%} probability")
            else:
                st.metric(label="Predicted Movement", value="DOWN", delta=f"{(1-probability):.2%} probability")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the prediction API: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")