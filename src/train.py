"""Train the final transaction classifier on all the synthetic data and save it."""

from pathlib import Path

import pandas as pd

from src.classifier import TransactionClassifier

import os
from pathlib import Path

DATA_PATH = Path("data/processed/transactions_clean.csv")
MODEL_PATH = Path("models/classifier.joblib")
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"

SANITY_CHECKS = [
    "CB CARREFOUR MARKET 12/03",
    "PRLV SEPA NETFLIX.COM",
    "RETRAIT DAB 05/04 PARIS 15",
    "VIR SEPA RECU DE CAF",
]


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Training on {len(df)} labels (embeddings: {USE_EMBEDDINGS})...")
    classifier = TransactionClassifier(use_embeddings=USE_EMBEDDINGS).fit(df["label"], df["category"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    classifier.save(MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1_000_000
    print(f"Saved to {MODEL_PATH} ({size_mb:.1f} MB)\n")

    loaded = TransactionClassifier.load(MODEL_PATH)
    for label, (category, confidence) in zip(SANITY_CHECKS, loaded.predict(SANITY_CHECKS)):
        print(f"{label:<30} -> {category:<16} ({confidence:.0%})")


if __name__ == "__main__":
    main()