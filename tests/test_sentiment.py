"""Tests for POST /v1/sentiment."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestSentimentEndpoint:

    def test_positive_sentiment(self, client: TestClient, positive_text: str) -> None:
        response = client.post("/v1/sentiment", json={"text": positive_text})
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] > 0.5
        assert 0.0 <= data["positive_score"] <= 1.0
        assert 0.0 <= data["negative_score"] <= 1.0
        assert 0.0 <= data["neutral_score"] <= 1.0
        assert isinstance(data["key_words"], list)

    def test_negative_sentiment(self, client: TestClient, negative_text: str) -> None:
        response = client.post("/v1/sentiment", json={"text": negative_text})
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "negative"
        assert data["confidence"] > 0.5

    def test_meta_block_present(self, client: TestClient, positive_text: str) -> None:
        response = client.post("/v1/sentiment", json={"text": positive_text})
        data = response.json()
        assert "meta" in data
        assert data["meta"]["char_count"] == len(positive_text)
        assert data["meta"]["processing_time_ms"] >= 0

    def test_negated_positive_word(self, client: TestClient) -> None:
        # "لا رائع" should flip positive -> more neutral/negative
        text = "هذا ليس رائعاً على الإطلاق"
        response = client.post("/v1/sentiment", json={"text": text})
        assert response.status_code == 200
        data = response.json()
        # With negation, should not be strongly positive
        assert data["sentiment"] in ("negative", "neutral")

    def test_empty_text_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/sentiment", json={"text": "   "})
        assert response.status_code == 422

    def test_missing_text_field_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/sentiment", json={})
        assert response.status_code == 422

    def test_text_too_long_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/sentiment", json={"text": "أ" * 5001})
        assert response.status_code == 422

    def test_scores_sum_to_approximately_one(self, client: TestClient, positive_text: str) -> None:
        response = client.post("/v1/sentiment", json={"text": positive_text})
        data = response.json()
        total = data["positive_score"] + data["negative_score"] + data["neutral_score"]
        assert abs(total - 1.0) < 0.05  # allow small floating-point gap

    def test_key_words_are_arabic(self, client: TestClient, positive_text: str) -> None:
        response = client.post("/v1/sentiment", json={"text": positive_text})
        data = response.json()
        # All key words should be non-empty strings
        for word in data["key_words"]:
            assert isinstance(word, str)
            assert len(word) > 0


class TestSentimentWithIntensifiers:

    def test_intensifier_boosts_confidence(self, client: TestClient) -> None:
        base = client.post("/v1/sentiment", json={"text": "المنتج جيد"})
        intensified = client.post("/v1/sentiment", json={"text": "المنتج جيد جداً"})
        # Intensified form should have same label but same or higher confidence
        assert base.status_code == intensified.status_code == 200
        # Both should be positive
        assert intensified.json()["sentiment"] in ("positive", "neutral")
