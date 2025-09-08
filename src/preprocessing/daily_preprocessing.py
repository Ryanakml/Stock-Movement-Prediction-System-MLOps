import pandas as pd
import re
import string
from datetime import datetime
import os

# Stopwords custom (tanpa NLTK, aman buat CI/CD)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "them",
    "my", "your", "his", "their", "our", "this", "that", "these", "those",
    "of", "for", "to", "in", "on", "at", "by", "from", "up", "down", "out",
}

# Regex untuk hapus emoji & simbol Unicode
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticon
    "\U0001F300-\U0001F5FF"  # simbol & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U000024C2-\U0001F251"  # symbols misc
    "]+",
    flags=re.UNICODE,
)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca

    # Hapus emoji & simbol lain (selain huruf, angka, dan spasi)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Opsional: kadang masih ada unicode high-plane (emoji 🚀💸😉), filter lagi
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)

    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS]  # hapus stopwords
    tokens = [w[:-2] if len(w) > 3 else w for w in tokens]  # pseudo stemming
    return " ".join(tokens)



def preprocess_file(input_path: str, output_path: str):
    """Generic preprocessing untuk 1 file CSV."""
    if not os.path.exists(input_path):
        print(f"⚠️ Input file {input_path} not found, skip.")
        return

    df = pd.read_csv(input_path)

    # Preprocess kolom text
    if "title" in df.columns:
        df["title_clean"] = df["title"].apply(preprocess_text)
    if "description" in df.columns:
        df["description_clean"] = df["description"].apply(preprocess_text)
    if "body" in df.columns:  # misalnya reddit punya kolom 'body'
        df["body_clean"] = df["body"].apply(preprocess_text)

    # Simpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    # Ambil tanggal hari ini
    current_date = datetime.now().strftime("%Y-%m-%d")
  

    # News
    news_input = f"data/live/news/AAPL_news_data_{current_date}.csv"
    news_output = f"data/processed/news/processed_news_{current_date}.csv"
    preprocess_file(news_input, news_output)

    # Reddit
    reddit_input = f"data/live/reddit/AAPL_reddit_data_{current_date}.csv"
    reddit_output = f"data/processed/reddit/processed_reddit_{current_date}.csv"
    preprocess_file(reddit_input, reddit_output)
