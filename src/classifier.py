"""Transaction classifier: TF-IDF, optional sentence embeddings, and logistic regression."""

import joblib
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from src.preprocessing import clean_label

import json

from anthropic import Anthropic

ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

LLM_MODEL = "claude-haiku-5-5"

CATEGORY_DEFINITIONS = {
    "Groceries": "supermarkets, convenience stores, organic shops, frozen food stores, independent bakeries and butchers",
    "Restaurants": "restaurants, fast food, cafes, bakery-cafe chains, bars, food delivery apps",
    "Transport": "trains, public transport, taxis and ride-hailing, car rental, bike and scooter sharing, fuel, parking, tolls, EV charging",
    "Subscriptions": "digital subscriptions: video and music streaming, software, cloud storage, online news, AI tools",
    "Shopping": "retail stores and online shops: clothing, electronics, home, DIY, beauty, marketplaces",
    "Housing & Utilities": "rent, property agencies, electricity, gas, water, internet and mobile operators, home and car insurance, property tax",
    "Health": "pharmacies, doctors, dentists, medical labs, opticians, health insurance (mutuelles)",
    "Leisure & Travel": "cinemas, concerts, events, hotels, holiday rentals, airlines, video games, theme parks, museums, gyms, lottery",
    "Income": "money received from employers (salary), public bodies (CAF, CPAM, France Travail, tax refunds) and pension funds",
    "Transfers": "transfers to or from individuals, savings accounts, and payment apps like Lydia",
    "Cash Withdrawal": "cash withdrawals at ATMs or bank counters",
    "Bank Fees": "fees, commissions, card fees and interest charged by the bank",
}

TAXONOMY_RULES = [
    "Gyms and fitness clubs are Leisure & Travel, never Health.",
    "Video games, game stores and gaming subscriptions (Xbox, PlayStation, Nintendo, Steam) are Leisure & Travel, never Subscriptions or Shopping.",
    "Newspapers and magazines, including sports news, are Subscriptions.",
]

def build_llm_prompt(labels):
    categories = "\n".join(f"- {name}: {definition}" for name, definition in CATEGORY_DEFINITIONS.items())
    rules = "\n".join(f"- {rule}" for rule in TAXONOMY_RULES)
    numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, start=1))
    return f"""You categorize French bank transaction labels.

Categories:
{categories}

Rules that override intuition:
{rules}

Labels:
{numbered}

For each label, identify the merchant or counterparty, then choose the single best category.
Return only a JSON object mapping each label number to its category, like {{"1": "Groceries", "2": "Transport"}}, with no explanation and no markdown."""

def ask_claude(labels, client=None):
    """Categorize a batch of raw labels with Claude. Returns one category per label, or None if missing or invalid."""
    client = client or Anthropic()
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=2000,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": build_llm_prompt(labels)}],
    )
    if response.stop_reason == "max_tokens":
        raise ValueError("Claude's answer was cut off: send fewer labels per batch")

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    answers = json.loads(text)
    if not isinstance(answers, dict):
        raise ValueError("Claude did not return a JSON object")

    results = []
    for number in range(1, len(labels) + 1):
        category = answers.get(str(number))
        results.append(category if category in CATEGORY_DEFINITIONS else None)
    return results


class TransactionClassifier:
    """Categorize raw bank labels. Combines TF-IDF with optional sentence embeddings."""

    def __init__(self, use_embeddings=True, C=1.0):
        self.use_embeddings = use_embeddings
        self.C = C
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
        self.model = LogisticRegression(C=C, max_iter=1000)
        self._encoder = None

    @property
    def encoder(self):
        """Load the embedding model the first time it's needed."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(ENCODER_NAME)
        return self._encoder

    def _features(self, clean_texts, fit=False):
        if fit:
            tfidf = self.vectorizer.fit_transform(clean_texts)
        else:
            tfidf = self.vectorizer.transform(clean_texts)
        if not self.use_embeddings:
            return tfidf
        embeddings = self.encoder.encode([text.lower() for text in clean_texts], batch_size=64)
        return hstack([tfidf, csr_matrix(normalize(embeddings))]).tocsr()

    def fit(self, labels, categories):
        """Train on raw labels and their categories."""
        clean_texts = [clean_label(label) for label in labels]
        self.model.fit(self._features(clean_texts, fit=True), list(categories))
        return self

    def predict(self, labels):
        """Return one (category, confidence) pair per raw label."""
        clean_texts = [clean_label(label) for label in labels]
        probabilities = self.model.predict_proba(self._features(clean_texts))
        best = probabilities.argmax(axis=1)
        return [
            (self.model.classes_[index], float(probabilities[row, index]))
            for row, index in enumerate(best)
        ]

    def save(self, path):
        """Save to disk, without the embedding model (it's reloaded by name)."""
        encoder, self._encoder = self._encoder, None
        joblib.dump(self, path)
        self._encoder = encoder

    @staticmethod
    def load(path):
        return joblib.load(path)
    
CONFIDENCE_THRESHOLD = 0.7


def categorize(classifier, labels, threshold=CONFIDENCE_THRESHOLD, client=None, batch_size=25):
    """Categorize raw labels: the model decides when confident, Claude otherwise."""
    labels = list(labels)
    results = [
        {
            "label": label,
            "category": category,
            "confidence": confidence,
            "model_category": category,
            "source": "model",
        }
        for label, (category, confidence) in zip(labels, classifier.predict(labels))
    ]

    uncertain = [i for i, result in enumerate(results) if result["confidence"] < threshold]
    for start in range(0, len(uncertain), batch_size):
        positions = uncertain[start:start + batch_size]
        try:
            answers = ask_claude([labels[i] for i in positions], client=client)
        except Exception:
            continue
        for i, answer in zip(positions, answers):
            if answer is not None:
                results[i]["category"] = answer
                results[i]["source"] = "claude"

    return results
    
