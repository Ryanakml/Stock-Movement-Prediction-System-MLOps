# file: src/data/build_features.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

def combine_and_save_data(historical_path: str, daily_path: str, unique_subset: list):
    """
    Menggabungkan file data historis dengan file data harian yang baru.
    
    Jika file historis tidak ada, file harian akan disalin sebagai basis.
    Jika file harian tidak ada, proses akan dilewati.
    """
    # Cek apakah ada file data baru untuk diproses
    if not os.path.exists(daily_path):
        print(f"INFO: File data harian tidak ditemukan di {daily_path}. Proses penggabungan dilewati.")
        return

    # Jika file historis sudah ada, gabungkan keduanya
    if os.path.exists(historical_path):
        print(f"INFO: Menggabungkan data historis '{historical_path}' dengan data baru '{daily_path}'...")
        historical_df = pd.read_csv(historical_path)
        daily_df = pd.read_csv(daily_path)
        
        combined_df = pd.concat([historical_df, daily_df], ignore_index=True)
        
        # Hapus duplikat berdasarkan kolom unik dan simpan data yang paling baru
        combined_df.drop_duplicates(subset=unique_subset, keep='last', inplace=True)
        
        # Urutkan data berdasarkan tanggal untuk memastikan konsistensi
        date_col = next((col for col in ['Date', 'publishedAt', 'created_utc'] if col in combined_df.columns), None)
        if date_col:
            combined_df[date_col] = pd.to_datetime(combined_df[date_col], errors='coerce', utc=True if 'utc' in date_col else False)
            combined_df.sort_values(by=date_col, inplace=True)

        combined_df.to_csv(historical_path, index=False)
        print(f"INFO: Data berhasil digabungkan dan disimpan kembali ke {historical_path}")
    else:
        # Jika file historis belum ada, salin file baru sebagai file historis pertama
        print(f"INFO: File historis tidak ditemukan. Membuat file baru dari {daily_path}...")
        daily_df = pd.read_csv(daily_path)
        os.makedirs(os.path.dirname(historical_path), exist_ok=True)
        daily_df.to_csv(historical_path, index=False)

def aggregate_sentiment_scores(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Aggregates sentiment scores by day.
    """
    df[date_column] = pd.to_datetime(df[date_column], utc=True).dt.date
    
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_numeric'] = df['sentiment'].map(sentiment_map) * df['sentiment_score']
    
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
    current_date = datetime.today().strftime('%Y-%m-%d')
    
    #  LANGKAH 1: Definisikan Path untuk Data Historis dan Harian 
    # Path data historis (master files)
    tech_indicators_path = f"data/final/update_{ticker}_technical_indicators.csv"
    news_sentiment_path = f"data/final/update_{ticker}_news_sentiment.csv"
    reddit_sentiment_path = f"data/final/update_{ticker}_reddit_sentiment.csv"

    # Path data harian (file baru yang akan digabungkan)
    # Asumsi file harian disimpan di folder 'data/live' dengan format tanggal
    daily_tech_path = f"data/featured/technical/{ticker}_technical_indicators_{current_date}.csv"
    daily_news_path = f"data/featured/news/{ticker}_news_data_{current_date}_sentiment.csv" # Asumsi nama file hasil sentimen
    daily_reddit_path = f"data/featured/reddit/{ticker}_reddit_data_{current_date}_sentiment.csv" # Asumsi nama file hasil sentimen

    #  LANGKAH 2: Gabungkan Data Historis dengan Data Harian 
    print(" Memulai proses penggabungan data ")
    combine_and_save_data(tech_indicators_path, daily_tech_path, unique_subset=['Date'])
    combine_and_save_data(news_sentiment_path, daily_news_path, unique_subset=['title', 'publishedAt'])
    combine_and_save_data(reddit_sentiment_path, daily_reddit_path, unique_subset=['id']) # 'id' biasanya unik untuk reddit
    print(" Proses penggabungan data selesai \n")

    #  LANGKAH 3: Proses Pembuatan Dataset Final (Logika Lama) 
    print(" Memulai proses pembuatan dataset final ")
    if not os.path.exists(tech_indicators_path):
        raise FileNotFoundError(f"File indikator teknis tidak ditemukan: {tech_indicators_path}. Pastikan data sudah ada.")

    tech_df = pd.read_csv(tech_indicators_path, index_col='Date', parse_dates=True)

    # Aggregate dan merge news sentiment
    if os.path.exists(news_sentiment_path):
        news_df = pd.read_csv(news_sentiment_path)
        daily_news_sentiment = aggregate_sentiment_scores(news_df, 'publishedAt')
        tech_df = tech_df.join(daily_news_sentiment, how='left', rsuffix='_news')

    # Aggregate dan merge reddit sentiment
    if os.path.exists(reddit_sentiment_path):
        reddit_df = pd.read_csv(reddit_sentiment_path)
        daily_reddit_sentiment = aggregate_sentiment_scores(reddit_df, 'created_utc')
        tech_df = tech_df.join(daily_reddit_sentiment, how='left', rsuffix='_reddit')
        
    tech_df.fillna(method='ffill', inplace=True)
    tech_df.dropna(inplace=True) 

    tech_df['target'] = (tech_df['close'].shift(-1) > tech_df['close']).astype(int)
    tech_df.dropna(subset=['target'], inplace=True)
    
    output_path = f"data/processed/update_{ticker}_final_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tech_df.to_csv(output_path)
    print(f"Dataset final berhasil dibuat dan disimpan di {output_path}")
    print(tech_df.head())
    print(f"Bentuk dataset: {tech_df.shape}")
    print(f"Distribusi target:\n{tech_df['target'].value_counts(normalize=True)}")

if __name__ == '__main__':
    TICKER = "AAPL"
    create_final_dataset(TICKER)