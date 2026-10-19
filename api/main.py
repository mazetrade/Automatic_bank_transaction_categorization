"""REST API for the SpendSense transaction categorizer."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.classifier import CONFIDENCE_THRESHOLD, TransactionClassifier, categorize
from fastapi.responses import FileResponse

MODEL_PATH = Path("models/classifier.joblib")
MAX_LABELS = 100
CACHE_MAX_SIZE = 10_000
STATIC_DIR = Path(__file__).parent / "static"

load_dotenv()
state = {}


@asynccontextmanager
async def lifespan(app):
    state["classifier"] = TransactionClassifier.load(MODEL_PATH)
    state["classifier"].predict(["warm up"])
    state["client"] = Anthropic() if os.getenv("ANTHROPIC_API_KEY") else None
    state["cache"] = {}
    yield
    state.clear()


app = FastAPI(
    title="SpendSense",
    description="Categorizes French bank transaction labels with a hybrid model and a Claude fallback.",
    version="1.0.0",
    lifespan=lifespan,
)


class CategorizeRequest(BaseModel):
    labels: list[str] = Field(
        min_length=1,
        max_length=MAX_LABELS,
        examples=[["CB CARREFOUR MARKET 12/03", "PRLV SEPA NETFLIX.COM", "CB KINEPOLIS 30/03"]],
    )


class TransactionResult(BaseModel):
    label: str
    category: str
    confidence: float = Field(
        description="The model's confidence in its own prediction. When source is 'claude', "
        "this low confidence is the reason Claude was asked."
    )
    source: str = Field(description="Who made the final decision: 'model' or 'claude'.")


class CategorizeResponse(BaseModel):
    results: list[TransactionResult]

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_fallback": state.get("client") is not None,
        "cached_labels": len(state.get("cache", {})),
    }


@app.post("/categorize", response_model=CategorizeResponse)
def categorize_labels(request: CategorizeRequest):
    client = state["client"]
    threshold = CONFIDENCE_THRESHOLD if client else 0.0
    cache = state["cache"]

    new_labels = [label for label in dict.fromkeys(request.labels) if label not in cache]
    fresh = {}
    if new_labels:
        for result in categorize(state["classifier"], new_labels, threshold=threshold, client=client):
            fresh[result["label"]] = result
            reliable = result["source"] == "claude" or result["confidence"] >= threshold
            if reliable and len(cache) < CACHE_MAX_SIZE:
                cache[result["label"]] = result

    results = [fresh.get(label) or cache[label] for label in request.labels]
    return {"results": [{**result, "confidence": round(result["confidence"], 3)} for result in results]}