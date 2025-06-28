import os
import re
import requests
import pandas as pd
from typing import List

NEWS_API_URL = "https://newsdata.io/api/1/news"


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_news(query: str, language: str = "en", max_records: int = 100) -> pd.DataFrame:
    """Fetch news articles from newsdata.io."""
    api_key = os.environ.get("NEWSDATA_API_KEY")
    if not api_key:
        raise EnvironmentError("NEWSDATA_API_KEY not set")

    articles: List[dict] = []
    page = 0
    while len(articles) < max_records:
        params = {
            "apikey": api_key,
            "q": query,
            "language": language,
            "page": page + 1,
        }
        resp = requests.get(NEWS_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        for item in results:
            articles.append({
                "title": clean_text(item.get("title", "")),
                "description": clean_text(item.get("description", "")),
            })
            if len(articles) >= max_records:
                break
        if not data.get("nextPage"):
            break
        page += 1
    return pd.DataFrame(articles)


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = fetch_news(query="technology", max_records=50)
    save_data(df, "data/raw_news.csv")
    print(df.head())
