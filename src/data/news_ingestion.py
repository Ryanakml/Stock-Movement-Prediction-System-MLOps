import os
import requests
import pandas as pd
from dotenv import load_dotenv  # for accessing environment variables
from datetime import datetime, timedelta

load_dotenv()

# API Key Polygon.io
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY not found in environment variables")

# Chunking configuration
TICKER = "AAPL"
TOTAL_DAYS = 105
CHUNK_DAYS = 15

all_articles = []
print(f'Starting to fetch {TICKER} news articles for {TOTAL_DAYS} days')

# Loop fetch data for 105 days from now with 15 days chunk
for i in range(0, TOTAL_DAYS, CHUNK_DAYS):
    # Start day and end day for each chunk
    end_date = datetime.now() - timedelta(days=i)
    start_date = end_date - timedelta(days=CHUNK_DAYS)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    print(f'Fetching news for {TICKER} from {start_date} to {end_date}')

    # Endpoint for polygon api fetching
    url = (
        f"https://api.polygon.io/v2/reference/news?ticker={TICKER}"
        f"&published_utc.gte={start_date_str}&published_utc.lte={end_date_str}"
        f"&limit=100&apiKey={POLYGON_API_KEY}"
    )

    # Pegination, if more than 100 articles in 15 days
    while url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            articles_in_page = data.get('result', [])
            if not articles_in_page:
                print('There is no article more in this page')
                break

            all_articles.extend(articles_in_page)
            print(f'Found {len(articles_in_page)} articles in this page, Total Now : {len(all_articles)}')

            # Next url for pagination
            url = data.get('next_url')
            if url:
                url += f'&apiKey={POLYGON_API_KEY}'

            # Limit polygon (5 request per minutes)
            time.sleep(15)

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            url = None

# Create dataframe from data parsed
if all_articles:
    df = pd.DataFrame(all_articles)
    df = df[['published_utc', 'title', 'description', 'article_url']]
    df.rename(columns={'published_utc': 'publishedAt'}, inplace=True)
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_localize(None)

    # Save to csv
    DATA_PATH = 'data/raw/AAPL_news_data.csv'
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f'News articles for {TICKER} saved to {DATA_PATH}')
else:
    print(f'No articles fetched for {TICKER}')
