"""Brand-based train/test split and evaluation helpers."""

import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import clean_label

from sklearn.metrics import accuracy_score, f1_score

GENERIC_WORDS = {
    "LA", "LE", "LES", "L", "SAINT", "GRAND", "GRANDE", "SUPER", "HYPER",
    "CAFE", "BRASSERIE", "PHARMACIE", "CABINET", "CENTRE", "LABORATOIRE",
    "MUTUELLE", "ASSURANCE", "BANQUE", "CREDIT", "CAISSE", "AGENCE", "SCI", "HOTEL",
    "HOTELS", "CINEMA", "PARC", "MUSEE", "CARTE", "SUMUP", "ZETTLE",
    "FRAIS", "COTIS", "COTISATION", "COMMISSION", "INTERETS",
}


def brand_of(merchant):
    """Return the brand of a merchant: its first word, or its full name if the first word is generic."""
    words = clean_label(merchant).split()
    if not words:
        return ""
    if words[0] in GENERIC_WORDS:
        return " ".join(words)
    return words[0]


def split_by_brand(df, n_splits=5, seed=42):
    """Split a DataFrame into train and test sets so that no brand appears in both."""
    groups = df["merchant"].apply(brand_of)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["category"], groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

def evaluate(y_true, y_pred, name="model"):
    """Print accuracy and macro F1, and return them as a dictionary."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name:<32} accuracy: {accuracy:.1%} | macro F1: {macro_f1:.1%}")
    return {"model": name, "accuracy": accuracy, "macro_f1": macro_f1}

def cross_val_by_brand(fit_predict, df, n_splits=5, seed=42):
    """Cross-validate with brand-grouped folds. Returns the mean and standard deviation of macro F1."""
    groups = df["merchant"].apply(brand_of)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in splitter.split(df, df["category"], groups):
        y_pred = fit_predict(train_idx, val_idx)
        scores.append(f1_score(df["category"].iloc[val_idx], y_pred, average="macro"))
    return np.mean(scores), np.std(scores)