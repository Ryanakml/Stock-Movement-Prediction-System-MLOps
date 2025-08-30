import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# API Key Polygon.io
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") 
TICKER = "AAPL"

# Rentang waktu
start_date = "2025-08-01"  # YYYY-MM-DD
end_date = "2025-08-30"

# Endpoint untuk berita ticker dengan rentang waktu
url = (
    f"https://api.polygon.io/v2/reference/news?"
    f"ticker={TICKER}&limit=100&apiKey={POLYGON_API_KEY}"
    f"&published_utc.gte={start_date}&published_utc.lte={end_date}"
)

response = requests.get(url)
data = response.json()

if response.status_code == 200 and data.get("results"):
    df = pd.DataFrame(data["results"])
    df = df[['published_utc', 'title', 'description', 'article_url']]
    df.rename(columns={'published_utc': 'publishedAt'}, inplace=True)
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_localize(None)

    # Pastikan folder data/raw ada
    os.makedirs("data/raw", exist_ok=True)
    csv_path = f"data/raw/{TICKER}_news_data.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Successfully fetched {len(df)} news articles for {TICKER}")
    print(f"Saved to {csv_path}")
    print()
    print(df.head())
    
else:
    print(f"Failed to fetch news: {data}")