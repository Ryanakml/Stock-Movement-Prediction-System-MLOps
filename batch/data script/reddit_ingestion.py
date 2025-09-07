# file: src/data/reddit_ingestion.py
import os
import praw
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def fetch_reddit_data(client_id, client_secret, user_agent, query, subreddits, limit=100):
    """
    Fetches Reddit submissions from specified subreddits based on a query.

    Args:
        client_id (str): Reddit API client ID.
        client_secret (str): Reddit API client secret.
        user_agent (str): A unique user agent string.
        query (str): The search query (e.g., 'AAPL' or '$AAPL').
        subreddits (list): A list of subreddit names to search in.
        limit (int): The maximum number of submissions to fetch.

    Returns:
        pd.DataFrame: A DataFrame with submission data.
    """
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    
    posts = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        # Search for the query within the subreddit, sorting by relevance or top
        for submission in subreddit.search(query, sort='relevance', time_filter='year', limit=limit):
            posts.append({
                'created_utc': datetime.fromtimestamp(submission.created_utc),
                'title': submission.title,
                'selftext': submission.selftext,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'subreddit': sub
            })
    
    if not posts:
        print(f"No Reddit posts found for query '{query}'.")
        return pd.DataFrame()

    df = pd.DataFrame(posts)
    df = df.sort_values(by='created_utc', ascending=False)
    print(f"Successfully fetched {len(df)} Reddit posts for query '{query}'.")
    return df

if __name__ == '__main__':
    # Store credentials as environment variables
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        raise ValueError("Reddit API credentials not set in environment variables.")

    QUERY = "AAPL OR Apple"
    SUBREDDITS = ["stocks", "wallstreetbets", "investing"]
    DATA_PATH = "data/raw/AAPL_reddit_data.csv"
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    reddit_df = fetch_reddit_data(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, QUERY, SUBREDDITS)
    if not reddit_df.empty:
        reddit_df.to_csv(DATA_PATH, index=False)
        print(f"Reddit data saved to {DATA_PATH}")
        print(reddit_df.head())