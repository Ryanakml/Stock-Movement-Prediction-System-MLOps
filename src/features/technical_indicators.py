# file: src/features/technical_indicators.py
import pandas as pd
import pandas_ta as ta
import os

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a suite of technical indicators to the stock price DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.

    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """
    # Create a custom strategy for multiple indicators
    custom_strategy = ta.Strategy(
        name="MomoAndVolatility",
        description="RSI, MACD, Bollinger Bands, and SMA",
        ta=[
            {"kind": "sma", "length": 20},
            {"kind": "sma", "length": 50},
            {"kind": "rsi"},
            {"kind": "macd", "fast": 12, "slow": 26},
            {"kind": "bbands", "length": 20},
        ]
    )
    
    # Apply the strategy to the DataFrame
    df.ta.strategy(custom_strategy)
    
    # Drop rows with NaN values created by indicators with lookback periods
    df.dropna(inplace=True)
    
    print("Technical indicators added successfully.")
    return df

if __name__ == "__main__":
    TICKER = "AAPL"
    INPUT_PATH = f"data/raw/{TICKER}_price_data.csv"
    OUTPUT_PATH = f"data/processed/{TICKER}_technical_indicators.csv"

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Price data not found at {INPUT_PATH}. Please run price_ingestion.py first.")
    else:
        # Baca CSV dengan Date sebagai index
        price_df = pd.read_csv(INPUT_PATH)

        # Kalau ada kolom Date → jadikan index
        if "Date" in price_df.columns:
            price_df["Date"] = pd.to_datetime(price_df["Date"])
            price_df.set_index("Date", inplace=True)
        else:
            # Kalau sudah jadi index → pastikan tipe datetime
            price_df.index = pd.to_datetime(price_df.index)

        # Pastikan kolom lowercase (pandas_ta standar pakai 'open','high','low','close','volume')
        price_df.columns = [col.lower() for col in price_df.columns]

        # Convert kolom numerik
        numeric_cols = ["open","high","low","close","volume"]
        for col in numeric_cols:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

        # Tambahkan technical indicators
        indicators_df = add_technical_indicators(price_df)

        # Simpan hasil
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        indicators_df.to_csv(OUTPUT_PATH)

        print(f"[SUCCESS] Data with technical indicators saved to {OUTPUT_PATH}")
        print(indicators_df.head())