"""Automated tests for the SpendSense API. Claude is replaced by a fake: no real API calls."""

import pytest
from fastapi.testclient import TestClient

import api.main
import src.classifier

claude_calls = []


def fake_ask_claude(labels, client=None):
    claude_calls.append(list(labels))
    return ["Leisure & Travel"] * len(labels)


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
        patch.setattr(src.classifier, "ask_claude", fake_ask_claude)
        with TestClient(api.main.app) as test_client:
            yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["llm_fallback"] is True


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SpendSense" in response.text


def test_known_label_is_decided_by_the_model(client):
    response = client.post("/categorize", json={"labels": ["CB CARREFOUR MARKET 12/03"]})
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["category"] == "Groceries"
    assert result["source"] == "model"
    assert 0.7 <= result["confidence"] <= 1


def test_unknown_label_goes_to_claude(client):
    response = client.post("/categorize", json={"labels": ["CB XQZWVBLORP 12/03"]})
    result = response.json()["results"][0]
    assert result["source"] == "claude"
    assert result["category"] == "Leisure & Travel"


def test_order_and_duplicates_are_preserved(client):
    labels = ["PRLV SEPA NETFLIX.COM", "RETRAIT DAB 05/04 PARIS 15", "PRLV SEPA NETFLIX.COM"]
    results = client.post("/categorize", json={"labels": labels}).json()["results"]
    assert [result["label"] for result in results] == labels
    assert results[0] == results[2]


def test_confidence_is_rounded(client):
    result = client.post("/categorize", json={"labels": ["CB PICARD 22/03"]}).json()["results"][0]
    assert result["confidence"] == round(result["confidence"], 3)


def test_cache_avoids_repeated_claude_calls(client):
    label = "CB ZORBLAXIUM 02/02"
    calls_before = len(claude_calls)
    client.post("/categorize", json={"labels": [label]})
    client.post("/categorize", json={"labels": [label]})
    assert len(claude_calls) == calls_before + 1


def test_empty_list_is_rejected(client):
    response = client.post("/categorize", json={"labels": []})
    assert response.status_code == 422


def test_too_many_labels_are_rejected(client):
    response = client.post("/categorize", json={"labels": ["CB CARREFOUR"] * 101})
    assert response.status_code == 422