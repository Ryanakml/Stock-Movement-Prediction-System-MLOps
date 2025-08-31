# file: ui/app.py
import streamlit as st
import requests
import pandas as pd
import os

# This would be a utility function to get the latest 30 days of features
# In a real app, this would involve running the full ingestion and feature engineering pipeline
def get_latest_features():
    # For this demo, we will load a pre-saved sample of scaled features
    # In a production system, this function would be much more complex
    # It would fetch live data, calculate indicators and sentiment, and scale them
    st.warning("Using pre-saved sample data for demonstration purposes.")
    sample_data = pd.read_csv("data/processed/AAPL_final_dataset.csv", index_col='Date', parse_dates=True)
    
    # We need to scale it first
    scaler = joblib.load("models/AAPL_scaler.joblib")
    feature_columns = sample_data.columns.drop('target')
    scaled_features = scaler.transform(sample_data[feature_columns])
    
    # Get the last 30 days
    latest_sequence = scaled_features[-30:]
    return latest_sequence.tolist()


# --- Streamlit App ---
st.title("Stock Price Movement Prediction")

# Get the API URL from an environment variable for flexibility
API_URL = os.getenv("API_URL", "YOUR_RENDER_API_URL_HERE")
if API_URL == "YOUR_RENDER_API_URL_HERE":
    st.error("Please set the API_URL environment variable.")

st.write("This dashboard predicts the next day's stock price movement for AAPL (Apple Inc.).")
st.write("Prediction: 1 for UP, 0 for DOWN.")

if st.button("Get Latest Prediction"):
    with st.spinner("Fetching latest data and making prediction..."):
        try:
            # 1. Get the latest feature data
            features = get_latest_features()
            
            # 2. Call the prediction API
            response = requests.post(API_URL, json={"data": features})
            response.raise_for_status() # Raise an exception for bad status codes
            
            result = response.json()
            
            # 3. Display the result
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