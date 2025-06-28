import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    df = df.dropna(subset=["description", "sentiment"])
    df["label"] = df["sentiment"].map(LABEL2ID)
    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(valid_df)


def tokenize_dataset(dataset: Dataset, tokenizer: DistilBertTokenizerFast) -> Dataset:
    return dataset.map(lambda x: tokenizer(x["description"], truncation=True, padding="max_length"), batched=True)


def train_model(train_ds: Dataset, valid_ds: Dataset, output_dir: str = "models/distilbert-sentiment") -> None:
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_ds = tokenize_dataset(train_ds, tokenizer)
    valid_ds = tokenize_dataset(valid_ds, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train_ds, valid_ds = load_dataset("data/annotated_news.csv")
    train_model(train_ds, valid_ds)
