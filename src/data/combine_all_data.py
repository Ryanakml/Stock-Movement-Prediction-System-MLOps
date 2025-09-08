# file: src/data/build_features.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

def combine_and_save_data(historical_path: str, daily_path: str, unique_subset: list):
    """
    Combine historical data with the new one, after preprocessed and featured.
    
    If historical file doesn't exist, daily file will be copied as the base.
    If daily file doesn't exist, the process will be skipped.
    If historical file doesn't exist, the process will create a new file from daily file.
    
    Args:
        historical_path (str): Path to the historical data file.
        daily_path (str): Path to the daily data file.
        unique_subset (list): List of columns to use for deduplication.
    """

    # Check if daily file exists
    if not os.path.exists(daily_path):
        print(f"INFO: Daily data is not found at: {daily_path}. The process will be skipped.")
        return

    # If historical file exists, combine both
    if os.path.exists(historical_path):
        print(f"INFO: Combining historical data at: {historical_path} with daily data at: {daily_path}...")
        historical_df = pd.read_csv(historical_path)
        daily_df = pd.read_csv(daily_path)
        
        combined_df = pd.concat([historical_df, daily_df], ignore_index=True)
        
        # Delete duplicates based on unique_subset
        combined_df.drop_duplicates(subset=unique_subset, keep='last', inplace=True)
        
        # Sort data by date to ensure consistency
        date_col = next((col for col in ['Date', 'publishedAt', 'created_utc'] if col in combined_df.columns), None)
        if date_col:
            combined_df[date_col] = pd.to_datetime(combined_df[date_col], errors='coerce', utc=True if 'utc' in date_col else False)
            combined_df.sort_values(by=date_col, inplace=True)

        combined_df.to_csv(historical_path, index=False)
        print(f"INFO: Data successfully combined and saved back to: {historical_path}")
    else:
        # If historical file doesn't exist, copy daily file as the base
        print(f"INFO: Historical data is not found. Creating a new file from: {daily_path}...")
        daily_df = pd.read_csv(daily_path)
        os.makedirs(os.path.dirname(historical_path), exist_ok=True)
        daily_df.to_csv(historical_path, index=False)

def aggregate_sentiment_scores(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Aggregates sentiment scores by day.
    """
    df[date_column] = pd.to_datetime(df[date_column], utc=True).dt.date
    
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_numeric'] = df['sentiment'].map(sentiment_map) * df['sentiment_score']
    
    daily_sentiment = df.groupby(date_column).agg(
        avg_sentiment_score=('sentiment_numeric', 'mean'),
        num_articles=('title', 'count'),
        positive_ratio=('sentiment', lambda x: (x == 'positive').sum() / len(x)),
        negative_ratio=('sentiment', lambda x: (x == 'negative').sum() / len(x))
    ).reset_index()
    
    daily_sentiment[date_column] = pd.to_datetime(daily_sentiment[date_column])
    daily_sentiment.set_index(date_column, inplace=True)
    
    return daily_sentiment

def create_final_dataset(ticker: str):
    """
    Merges technical indicators and sentiment data, and creates the target variable.
    """
    current_date = datetime.today().strftime('%Y-%m-%d')
    
    # 1. Define paths for historical and daily data
    # Historical data paths
    tech_indicators_path = f"data/final/{ticker}_technical_indicators.csv"
    news_sentiment_path = f"data/final/{ticker}_news_sentiment.csv"
    reddit_sentiment_path = f"data/final/{ticker}_reddit_sentiment.csv"

    # New data file path (The one to combine to)
    daily_tech_path = f"data/featured/price/technical_indicators_{current_date}.csv"
    daily_news_path = f"data/featured/news/news{current_date}.csv"
    daily_reddit_path = f"data/featured/reddit/reddit{current_date}.csv"

    # 2. Combine historical data with daily data
    print("Starting combining historical data with daily data")
    combine_and_save_data(tech_indicators_path, daily_tech_path, unique_subset=['Date'])
    combine_and_save_data(news_sentiment_path, daily_news_path, unique_subset=['title', 'publishedAt'])
    combine_and_save_data(reddit_sentiment_path, daily_reddit_path, unique_subset=['id']) # 'id' biasanya unik untuk reddit
    print(" Proses penggabungan data selesai \n")

    # 3. Create final dataset
    print("Merge all data into single entity (final dataset)")
    if not os.path.exists(tech_indicators_path):
        raise FileNotFoundError(f"Price data not found at: {tech_indicators_path}")

    tech_df = pd.read_csv(tech_indicators_path, index_col='Date', parse_dates=True)

    # Aggregate and merge news sentiment
    if os.path.exists(news_sentiment_path):
        news_df = pd.read_csv(news_sentiment_path)
        daily_news_sentiment = aggregate_sentiment_scores(news_df, 'publishedAt')
        tech_df = tech_df.join(daily_news_sentiment, how='left', rsuffix='_news')

    # Aggregate and merge reddit sentiment
    if os.path.exists(reddit_sentiment_path):
        reddit_df = pd.read_csv(reddit_sentiment_path)
        daily_reddit_sentiment = aggregate_sentiment_scores(reddit_df, 'created_utc')
        tech_df = tech_df.join(daily_reddit_sentiment, how='left', rsuffix='_reddit')
        
    tech_df.fillna(method='ffill', inplace=True)
    tech_df.dropna(inplace=True) 

    tech_df['target'] = (tech_df['close'].shift(-1) > tech_df['close']).astype(int)
    tech_df.dropna(subset=['target'], inplace=True)
    
    output_path = f"data/final/{ticker}_final_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tech_df.to_csv(output_path)
    print(f"Final dataset created and saved to {output_path}")
    print(tech_df.head())
    print(f"Dataset shape: {tech_df.shape}")
    print(f"Target distribution:\n{tech_df['target'].value_counts(normalize=True)}")

if __name__ == '__main__':
    TICKER = "AAPL"
    create_final_dataset(TICKER)