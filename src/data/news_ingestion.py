# file: src/data/news_ingestion.py
import os
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def fetch_news_data(api_key: str, query: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches news articles related to a query within a date range.

    Args:
        api_key (str): Your NewsAPI key.
        query (str): The search query (e.g., 'Apple' or 'AAPL').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with news articles.
    """
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = []
    
    # NewsAPI's 'everything' endpoint on the free plan is limited to the last month.
    # For a portfolio project, we can fetch recent data or use a sample historical dataset.
    # Here, we fetch data for the last 28 days for demonstration.
    
    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    
    # To respect API limits, we fetch data in chunks if needed, though for a short period it's not necessary.
    try:
        response = newsapi.get_everything(
            q=query,
            from_param=start.strftime('%Y-%m-%d'),
            to=end.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt',
            page_size=100 # Max page size
        )
        if response.get("status") != "ok":
            raise ValueError(f"API error: {response}")
        articles = response.get("articles", [])
        all_articles.extend(articles)
        
        df = pd.DataFrame(all_articles)
        if not df.empty:
            df = df[['publishedAt', 'title', 'description', 'content']]
            df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_localize(None)
            print(f"Successfully fetched {len(df)} news articles for query '{query}'.")
            return df
        else:
            print(f"No news articles found for query '{query}'.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"An error occurred during news fetching: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # IMPORTANT: Store your API key as an environment variable, not in the code.
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY environment variable not set.")

    QUERY = "Apple OR AAPL"
    END_DATE_DT = datetime.now()
    START_DATE_DT = END_DATE_DT - timedelta(days=28) # Free tier limit
    
    DATA_PATH = f"data/raw/AAPL_news_data.csv"
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    news_df = fetch_news_data(NEWS_API_KEY, QUERY, START_DATE_DT.strftime('%Y-%m-%d'), END_DATE_DT.strftime('%Y-%m-%d'))
    if not news_df.empty:
        news_df.to_csv(DATA_PATH, index=False)
        print(f"News data saved to {DATA_PATH}")
        print(news_df.head())