# file: src/data/live/news_ingestion_daily.py
import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# API Key Polygon.io
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") 
TICKER = "AAPL"
BASE_URL = "https://api.polygon.io/v2/reference/news"

def ingest_daily_news(ticker: str, date: str, limit: int = 10) -> pd.DataFrame:
    """
    Ingest daily news for a given ticker from Polygon.io.
    This will run once per day via orchestrator/CI-CD pipeline.

    Args:
        ticker (str): Stock ticker symbol.
        date (str): Date in format 'YYYY-MM-DD'.
        limit (int): Maximum number of news articles to fetch.

    Returns:
        pd.DataFrame: DataFrame containing daily news.
    """
    try:
        url = (
            f"{BASE_URL}?ticker={ticker}"
            f"&published_utc.gte={date}"
            f"&published_utc.lte={date}"
            f"&limit={limit}&apiKey={POLYGON_API_KEY}"
        )

        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            raise ValueError(f"Request failed: {data}")

        if not data.get("results"):
            print(f"No news available for {ticker} on {date}")
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        df = df[['published_utc', 'title', 'description', 'article_url']]
        df.rename(columns={'published_utc': 'publishedAt'}, inplace=True)
        df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_localize(None)

        print(f"Successfully ingested {len(df)} news articles for {ticker} on {date}")
        return df

    except Exception as e:
        print(f"Error ingesting news for {ticker} on {date}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Ensure folder exists
    os.makedirs("data/live/news", exist_ok=True)

    # Fetch daily news
    news_df = ingest_daily_news(TICKER, current_date, limit=10)

    if not news_df.empty:
        csv_path = f"data/live/news/{TICKER}_news_data_{current_date}.csv"
        news_df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")
        print(news_df.head())
    else:
        print(f"No news data saved for {TICKER} on {current_date}")
