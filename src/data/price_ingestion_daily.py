# file : src/data/live/price_ingestion_daily.py
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def ingest_price_data(ticker: str, current_date: str) -> pd.DataFrame:
    """
    Ingest price data daily, this will run automatically for every day on orchestrator,
    and being set to run on schedule everyday on CI/CD workflow.

    Args:
        ticker (str): Ticker symbol of the stock.
        current_date (str): Current date in format 'YYYY-MM-DD'.
        auto_adjust (bool): Whether to adjust the stock data automatically.
        progress (bool): Whether to show progress bar.

    Returns:
        pd.DataFrame: DataFrame containing price data for the specified ticker and date.
    """

    try:
        stock_data = yf.download(
            ticker,
            auto_adjust=True,
            progress=False
        )

        if stock_data.empty:
            raise ValueError(f"No data found for {ticker} on {current_date}")
        
        # Ensure the index is DatatimeIndex and timezone aware
        stock_data.index = pd.to_datetime(stock_data.index).tz_localize('UTC')

        print(f'Successfully ingested price data for {ticker} on {current_date}')
        return stock_data
    
    except Exception as e:
        print(f'Error ingesting price data for {ticker} on {current_date}: {e}')
        return pd.DataFrame()

if __name__ == '__main__':
    ticker = 'AAPL'
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Ensure directory exist
    os.makedirs('data/live', exist_ok=True)

    # fetch and save data
    price_df = ingest_price_data(ticker, current_date)
    if not price_df.empty:
        # Save to CSV
        price_df.to_csv(f'data/live/price/{ticker}_price_data_{current_date}.csv')
    else:
        print(f'No price data available for {ticker} on {current_date}')
    print(f'Price Data on {current_date}:\n{price_df.tail()}')
