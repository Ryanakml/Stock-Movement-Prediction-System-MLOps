# file: src/data/price_ingestion.py
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_price_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetches historical stock price data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        interval (str): The data interval. Valid intervals: 1d, 5d, 1wk, 1mo, 3mo.
        auto_adjust (bool): Whether to adjust for stock splits and dividends.
        progress (bool): Progress bar for status.
    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data, or an empty DataFrame if fetching fails.
    """

    try:
        stock_data = yf.download(
            tickers=ticker,   
            start=start_date, 
            end=end_date,     
            interval=interval,
            auto_adjust=True, 
            progress=False    
        )
        
        if stock_data.empty:
            print(f"No data found for {ticker} from {start_date} to {end_date}.")
            return pd.DataFrame()
        
        # Ensure the index is a DatetimeIndex and timezone-naive
        stock_data.index = pd.to_datetime(stock_data.index).tz_localize(None)
        
        print(f"Successfully fetched {len(stock_data)} data points for {ticker}.")
        return stock_data
        
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Configuration
    TICKER = "AAPL"
    START_DATE = "2024-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    DATA_PATH = f"data/raw/{TICKER}_price_data.csv"

    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Fetch and save data
    price_df = fetch_price_data(TICKER, START_DATE, END_DATE)
    if not price_df.empty:
        price_df.to_csv(DATA_PATH)
        print(f"Data saved to {DATA_PATH}")
        print(price_df.head())