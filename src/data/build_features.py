# file: src/data/build_features.py
import pandas as pd
import numpy as np
import os

def aggregate_sentiment_scores(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Aggregates sentiment scores by day.
    """
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    
    # Convert sentiment labels to numerical values for aggregation
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_numeric'] = df['sentiment'].map(sentiment_map) * df['sentiment_score']
    
    # Aggregate by date
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
    # Load data
    tech_indicators_path = f"data/processed/{ticker}_technical_indicators.csv"
    news_sentiment_path = f"data/processed/{ticker}_news_sentiment.csv"
    reddit_sentiment_path = f"data/processed/{ticker}_reddit_sentiment.csv"

    if not os.path.exists(tech_indicators_path):
        raise FileNotFoundError(f"Technical indicators file not found: {tech_indicators_path}")

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
        
    # Handle missing sentiment data (e.g., weekends, holidays) by forward-filling
    tech_df.fillna(method='ffill', inplace=True)
    tech_df.dropna(inplace=True) # Drop any remaining NaNs at the beginning

    # Create the target variable: 1 if next day's close is higher, else 0
    tech_df['target'] = (tech_df['close'].shift(-1) > tech_df['close']).astype(int)
    
    # Drop the last row as it will have a NaN target
    tech_df.dropna(subset=['target'], inplace=True)
    
    output_path = f"data/processed/{ticker}_final_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tech_df.to_csv(output_path)
    print(f"Final dataset created and saved to {output_path}")
    print(tech_df.head())
    print(f"Dataset shape: {tech_df.shape}")
    print(f"Target distribution:\n{tech_df['target'].value_counts(normalize=True)}")

if __name__ == '__main__':
    TICKER = "AAPL"
    create_final_dataset(TICKER)