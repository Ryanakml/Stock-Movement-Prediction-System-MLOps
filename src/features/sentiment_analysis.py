# file: src/features/sentiment_analysis.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import pandas as pd
from transformers.pipelines import pipeline
import torch


def analyze_sentiment(texts: list) -> list:
    """
    Performs sentiment analysis on a list of texts using FinBERT.

    Args:
        texts (list): A list of strings to analyze.

    Returns:
        list: A list of dictionaries, each containing the label ('positive', 'negative', 'neutral') and score.
    """
    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    
    # Using a specific, well-regarded FinBERT model from Hugging Face Hub
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="ProsusAI/finbert", 
        device=device
    )
    
    # The pipeline can process a list of texts directly
    # Truncate long texts to fit within the model's max sequence length
    results = sentiment_pipeline(texts, truncation=True, padding=True, max_length=512)
    return results

def process_sentiment_for_source(input_path: str, output_path: str, text_column: str):
    """
    Reads raw data, performs sentiment analysis, and saves the results.
    """
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    df = df.dropna(subset=[text_column])
    df[text_column] = df[text_column].astype(str)
    
    # Filter out empty strings
    texts_to_analyze = df[df[text_column].str.strip()!= ''][text_column].tolist()
    
    if not texts_to_analyze:
        print(f"No valid text found in {input_path} for column {text_column}.")
        return

    print(f"Analyzing sentiment for {len(texts_to_analyze)} texts from {input_path}...")
    sentiments = analyze_sentiment(texts_to_analyze)
    
    # Create a temporary DataFrame for sentiments to merge back
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_df.rename(columns={'label': 'sentiment', 'score': 'sentiment_score'}, inplace=True)
    
    # Align indices for merging
    valid_text_df = df[df[text_column].str.strip()!= ''].copy()
    valid_text_df.reset_index(drop=True, inplace=True)
    sentiment_df.reset_index(drop=True, inplace=True)
    
    result_df = pd.concat([valid_text_df, sentiment_df], axis=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Sentiment analysis complete. Results saved to {output_path}")

if __name__ == '__main__':
    # Process News Data
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    process_sentiment_for_source(
        input_path=f"data/raw/{ticker}_news_data.csv",
        output_path=f"data/processed/{ticker}_news_sentiment_{current_date}.csv",
        text_column='title' # Using title for news as it's more concise
    )
    
    # Process Reddit Data
    process_sentiment_for_source(
        input_path="data/raw/AAPL_reddit_data.csv",
        output_path=f"data/processed/AAPL_reddit_sentiment_{current_date}.csv",
        text_column='title' # Using title for Reddit posts as well
    )