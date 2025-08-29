# End-to-End MLOps: Stock Movement Prediction with Sentiment Analysis

This project implements a full, end-to-end MLOps pipeline to predict daily stock price movement (UP/DOWN) by combining technical indicators with financial news sentiment.

## Project Overview

The system ingests stock price data and financial news, engineers features (technical indicators and FinBERT-based sentiment scores), trains an LSTM model, and serves predictions via a REST API. The entire pipeline is automated with CI/CD, and features a closed-loop monitoring and retraining system that detects data drift and automatically retrains and redeploys the model.

### Live Demo

- **Prediction API (Render):**
    
- **User Dashboard (Hugging Face):**
    

## MLOps Architecture

!(path/to/your/architecture_diagram.png)

**Architectural Flow:**

1. **Data Ingestion:** Scheduled jobs pull raw stock prices from Yahoo Finance and textual data from News API and Reddit.
    
2. **Feature Engineering:** The raw data is processed. Technical indicators are calculated from price data, and sentiment scores are extracted from text data using a FinBERT model. These features are conceptually stored for retrieval.
    
3. **Training Pipeline:** An orchestrated workflow retrieves the latest features, trains a new model (e.g., LSTM), validates its performance, and versions the resulting artifact in a model registry.
    
4. **CI/CD:** Any code changes pushed to the main branch of the GitHub repository trigger an automated workflow that runs tests and, upon success, deploys the updated application.
    
5. **Serving & UI:** The trained model is served via a FastAPI application deployed on Render. A user-facing Streamlit dashboard, hosted on Hugging Face Spaces, interacts with this API to display predictions.
    
6. **Monitoring & Retraining:** A separate, scheduled process continuously monitors the live prediction data for drift against a reference dataset. If significant drift is detected, it automatically triggers the training pipeline to retrain the model on new data, thus closing the loop.
    

## Technology Stack

|MLOps Stage|Tool/Service|
|---|---|
|Data Ingestion|`yfinance`, `NewsAPI`, `PRAW`|
|Feature Eng.|`pandas-ta`, `Transformers`|
|Model|`TensorFlow/Keras` (LSTM)|
|API Serving|`FastAPI`, `Docker`|
|Deployment|`Render`, `Hugging Face Spaces`|
|CI/CD|`GitHub Actions`|
|Monitoring|`Evidently AI`|
|Orchestration|`Prefect`|

## [Detail Project](main.ipynb)..

## Setup and Installation

1. **Clone the repository:**bash

```bash    
git clone https://github.com/Ryanakml/Stock-Movement-Prediction-System-MLOps
cd Stock-Movement-Prediction-System-MLOps
```

2. Set up environment variables:

Create Virtual Environment :

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

Install Dependencies :

```bash
pip install -r requirements.txt
```
Add API Keys to ./bin/activate :

```bash
NEWS_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
RENDER_DEPLOY_HOOK_URL=...
```

3. Run the pipeline locally:
    
(Provide instructions on how to run the data ingestion, training, and local API server). 

Soon...