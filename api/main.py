from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import os

# Initialize FastAPI app
app = FastAPI(title="Stock Movement Prediction API")

# Load model and scaler artifacts
MODEL_DIR = "models"
TICKER = "AAPL"
try:
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{TICKER}_lstm_model.h5"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{TICKER}_scaler.joblib"))
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

# Define the input data model using Pydantic
class PredictionInput(BaseModel):
    # Expecting a list of lists representing (timesteps, features)
    # For a single prediction, this will be a list with one sequence.
    data: list[list[float]]
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API. Use the /predict endpoint for predictions."}

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Predicts stock movement based on a sequence of feature data.
    """
    try:
        # Convert input data to numpy array and reshape for the model
        input_array = np.array(input_data.data)
        if input_array.shape!= 30: # Assuming 30 timesteps
            raise ValueError(f"Input data must have 30 timesteps, but got {input_array.shape}")
            
        # Reshape for a single prediction: (timesteps, features) -> (1, timesteps, features)
        input_reshaped = np.expand_dims(input_array, axis=0)
        
        # Make prediction
        prediction_proba = model.predict(input_reshaped)
        prediction = 1 if prediction_proba > 0.5 else 0
        
        return {
            "prediction": prediction,
            "probability_up": float(prediction_proba)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")