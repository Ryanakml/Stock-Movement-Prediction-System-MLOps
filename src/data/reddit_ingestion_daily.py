import os
import praw
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

def fetch_reddit_data(client_id, client_secret, user_agent, query, subreddits, current_date: date = date.today()):
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    
    posts = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        for submission in subreddit.search(query, sort='new', time_filter='day', limit=10):
            submission_date = datetime.fromtimestamp(submission.created_utc).date()
            
            if submission_date == current_date:
                posts.append({
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'subreddit': sub
                })
    
    if not posts:
        print(f"No Reddit posts found for query '{query}' today.")
        return pd.DataFrame()

    df = pd.DataFrame(posts).sort_values(by='created_utc', ascending=False)
    print(f"Successfully fetched {len(df)} Reddit posts for query '{query}' today.")
    return df


if __name__ == '__main__':
    # Store credentials as environment variables
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        raise ValueError("Reddit API credentials not set in environment variables.")

    QUERY = "(AAPL OR Apple OR $AAPL OR 'AAPL stock' OR 'Apple stock' OR 'Apple earnings')"
    SUBREDDITS = ["stocks", "wallstreetbets", "investing", "StockMarket"]
    TICKER = "AAPL"
    current_date = datetime.now().strftime('%Y-%m-%d')
    DATA_PATH = f"data/live/reddit/{TICKER}_reddit_data_{current_date}.csv"
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Call function
    reddit_df = fetch_reddit_data(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, QUERY, SUBREDDITS)
    
    if not reddit_df.empty:
        reddit_df.to_csv(DATA_PATH, index=False)
        print(f"Reddit data saved to {DATA_PATH}")
        print(reddit_df.head())
        print(f'Total data fetched : {len(reddit_df)}')
    else:
        print("No Reddit data to save.")