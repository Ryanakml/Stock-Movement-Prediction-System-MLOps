import os
import sys
from datetime import datetime, timedelta
from prefect import flow, task
from prefect.schedules import IntervalSchedule
from prefect.task_runners import SequentialTaskRunner

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data ingestion modules
from src.data.price_ingestion_daily import ingest_price_data
from src.data.news_ingestion_daily import ingest_daily_news
from src.data.reddit_ingestion_daily import fetch_reddit_data

# Import feature engineering modules
from src.features.technical_indicators import add_technical_indicators
from src.features.sentiment_analysis import process_sentiment_for_source

# Import data combination and model training modules
from src.data.combine_all_data import create_final_dataset
from src.models.train_lstm import train_lstm_model

@task(name="Ingest Price Data", retries=3, retry_delay_seconds=60)
def price_ingestion_task(ticker: str):
    """Task to ingest daily price data"""
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_path = f"data/live/price/{ticker}_price_data_{current_date}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fetch and save data
    price_df = ingest_price_data(ticker, current_date)
    if not price_df.empty:
        price_df.to_csv(output_path)
        print(f"Price data saved to {output_path}")
        return output_path
    else:
        print(f"No price data available for {ticker} on {current_date}")
        return None

@task(name="Ingest News Data", retries=2, retry_delay_seconds=30)
def news_ingestion_task(ticker: str):
    """Task to ingest daily news data"""
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_path = f"data/live/news/{ticker}_news_data_{current_date}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fetch daily news
    news_df = ingest_daily_news(ticker, current_date, limit=10)
    if not news_df.empty:
        news_df.to_csv(output_path, index=False)
        print(f"News data saved to {output_path}")
        return output_path
    else:
        print(f"No news data saved for {ticker} on {current_date}")
        return None


@task(name="Ingest Reddit Data", retries=2, retry_delay_seconds=30)
def reddit_ingestion_task(ticker: str):
    """Task to ingest daily Reddit data"""
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_path = f"data/live/reddit/{ticker}_reddit_data_{current_date}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Reddit API credentials from environment variables
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")
    
    if not all([client_id, client_secret, user_agent]):
        print("Reddit API credentials not set in environment variables.")
        return None
    
    # Query and subreddits for Apple stock
    query = f"({ticker} OR Apple OR ${ticker} OR '{ticker} stock' OR 'Apple stock' OR 'Apple earnings')"
    subreddits = ["stocks", "wallstreetbets", "investing", "StockMarket"]
    
    # Fetch Reddit data
    reddit_df = fetch_reddit_data(client_id, client_secret, user_agent, query, subreddits)
    if not reddit_df.empty:
        reddit_df.to_csv(output_path, index=False)
        print(f"Reddit data saved to {output_path}")
        return output_path
    else:
        print("No Reddit data to save.")
        return None


@task(name="Generate Technical Indicators")
def technical_indicators_task(price_data_path: str):
    """Task to generate technical indicators from price data"""
    if not price_data_path or not os.path.exists(price_data_path):
        print(f"Price data not found at {price_data_path}")
        return None
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    ticker = os.path.basename(price_data_path).split('_')[0]
    output_path = f"data/featured/technical/{ticker}_technical_indicators_{current_date}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read price data
    price_df = pd.read_csv(price_data_path)
    
    # Process for technical indicators
    if "Date" in price_df.columns:
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df.set_index("Date", inplace=True)
    else:
        price_df.index = pd.to_datetime(price_df.index)
    
    # Ensure column names are lowercase
    price_df.columns = [col.lower() for col in price_df.columns]
    
    # Convert columns to numeric
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    
    # Add technical indicators
    indicators_df = add_technical_indicators(price_df)
    
    # Save results
    indicators_df.to_csv(output_path)
    print(f"Technical indicators saved to {output_path}")
    return output_path


@task(name="Process News Sentiment")
def news_sentiment_task(news_data_path: str):
    """Task to process sentiment from news data"""
    if not news_data_path or not os.path.exists(news_data_path):
        print(f"News data not found at {news_data_path}")
        return None
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    ticker = os.path.basename(news_data_path).split('_')[0]
    output_path = f"data/featured/news/{ticker}_news{current_date}_sentiment.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process sentiment
    process_sentiment_for_source(news_data_path, output_path, text_column='title')
    print(f"News sentiment saved to {output_path}")
    return output_path


@task(name="Process Reddit Sentiment")
def reddit_sentiment_task(reddit_data_path: str):
    """Task to process sentiment from Reddit data"""
    if not reddit_data_path or not os.path.exists(reddit_data_path):
        print(f"Reddit data not found at {reddit_data_path}")
        return None
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    ticker = os.path.basename(reddit_data_path).split('_')[0]
    output_path = f"data/featured/reddit/{ticker}_reddit{current_date}_sentiment.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process sentiment
    process_sentiment_for_source(reddit_data_path, output_path, text_column='title')
    print(f"Reddit sentiment saved to {output_path}")
    return output_path


@task(name="Combine All Data")
def combine_data_task(ticker: str):
    """Task to combine all processed data into a final dataset"""
    create_final_dataset(ticker)
    final_dataset_path = f"data/final/{ticker}_final_dataset.csv"
    if os.path.exists(final_dataset_path):
        print(f"Final dataset created at {final_dataset_path}")
        return final_dataset_path
    else:
        print("Failed to create final dataset")
        return None

@task(name="Train LSTM Model")
def train_model_task(ticker: str, time_steps: int = 5):
    """Task to train the LSTM model"""
    try:
        train_lstm_model(ticker, time_steps)
        model_path = f"models/{ticker}_lstm_model.h5"
        scaler_path = f"models/{ticker}_scaler.joblib"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"Model trained and saved to {model_path}")
            return model_path
        else:
            print("Model training failed")
            return None
    except Exception as e:
        print(f"Error training model: {e}")
        return None

@flow(name="Stock Prediction Pipeline", task_runner=SequentialTaskRunner())
def stock_prediction_pipeline(ticker: str = "AAPL"):
    """Main flow that orchestrates the entire stock prediction pipeline"""
    print(f"Starting stock prediction pipeline for {ticker} at {datetime.now()}")
    
    # Data ingestion tasks
    price_path = price_ingestion_task(ticker)
    news_path = news_ingestion_task(ticker)
    reddit_path = reddit_ingestion_task(ticker)
    
    # Feature engineering tasks - only run if data is available
    tech_indicators_path = None
    if price_path:
        tech_indicators_path = technical_indicators_task(price_path)
    
    news_sentiment_path = None
    if news_path:
        news_sentiment_path = news_sentiment_task(news_path)
    
    reddit_sentiment_path = None
    if reddit_path:
        reddit_sentiment_path = reddit_sentiment_task(reddit_path)
    
    # Only proceed with combination if we have at least technical indicators
    if tech_indicators_path:
        final_dataset_path = combine_data_task(ticker)
        
        # Only train model if we have a final dataset
        if final_dataset_path:
            model_path = train_model_task(ticker)
            if model_path:
                print(f"Pipeline completed successfully for {ticker}")
            else:
                print(f"Pipeline completed but model training failed for {ticker}")
        else:
            print(f"Pipeline completed but final dataset creation failed for {ticker}")
    else:
        print(f"Pipeline failed: No technical indicators generated for {ticker}")


# Schedule to run daily at 6:00 PM UTC (after market close)
schedule = IntervalSchedule(
    interval=timedelta(days=1),
    start_date=datetime.utcnow().replace(hour=18, minute=0, second=0, microsecond=0)
)

if __name__ == "__main__":
    # Import pandas here to avoid circular imports
    import pandas as pd
    
    # Run the pipeline
    stock_prediction_pipeline()