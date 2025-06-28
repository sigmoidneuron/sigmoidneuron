import os
import time
import pandas as pd
import openai
from typing import Iterable

openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY not set")


SYSTEM_PROMPT = (
    "You are a sentiment classifier. "
    "Given a news article, respond with one word: positive, neutral, or negative."
)


def annotate_sentiment(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )
    label = response.choices[0].message["content"].strip().lower()
    return label


def annotate_dataset(input_path: str, output_path: str, text_column: str = "description", delay: float = 1.0) -> None:
    df = pd.read_csv(input_path)
    labels: Iterable[str] = []
    for text in df[text_column].fillna(""):
        labels.append(annotate_sentiment(text))
        time.sleep(delay)
    df["sentiment"] = labels
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    annotate_dataset("data/raw_news.csv", "data/annotated_news.csv")
