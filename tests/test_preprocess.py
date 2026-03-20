"""Tests for POST /v1/preprocess."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestPreprocessEndpoint:

    def test_removes_diacritics(self, client: TestClient, diacritised_text: str) -> None:
        response = client.post(
            "/v1/preprocess",
            json={"text": diacritised_text, "remove_diacritics": True, "normalize": False},
        )
        assert response.status_code == 200
        data = response.json()
        # No harakat characters should remain
        import re
        harakat = re.compile(r"[\u064B-\u065F\u0610-\u061A\u0670]")
        assert not harakat.search(data["processed"])

    def test_normalises_alef_variants(self, client: TestClient) -> None:
        text = "إبراهيم أحمد آخر"  # إ أ آ should all become ا
        response = client.post(
            "/v1/preprocess",
            json={"text": text, "normalize": True, "remove_diacritics": False, "remove_punctuation": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "إ" not in data["processed"]
        assert "أ" not in data["processed"]
        assert "آ" not in data["processed"]

    def test_tokenize_returns_list(self, client: TestClient) -> None:
        response = client.post(
            "/v1/preprocess",
            json={"text": "الرجل يقرأ الكتاب", "tokenize": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["tokens"], list)
        assert data["token_count"] == len(data["tokens"])

    def test_tokenize_false_returns_empty_list(self, client: TestClient) -> None:
        response = client.post(
            "/v1/preprocess",
            json={"text": "الرجل يقرأ الكتاب", "tokenize": False},
        )
        data = response.json()
        assert data["tokens"] == []
        assert data["token_count"] == 0

    def test_operations_applied_logged(self, client: TestClient, diacritised_text: str) -> None:
        response = client.post(
            "/v1/preprocess",
            json={
                "text": diacritised_text,
                "normalize": True,
                "remove_diacritics": True,
                "remove_punctuation": True,
                "tokenize": True,
            },
        )
        data = response.json()
        ops = data["operations_applied"]
        assert "normalize_letters" in ops
        assert "remove_diacritics" in ops
        assert "remove_punctuation" in ops
        assert "tokenize" in ops

    def test_original_text_preserved(self, client: TestClient, diacritised_text: str) -> None:
        response = client.post("/v1/preprocess", json={"text": diacritised_text})
        data = response.json()
        assert data["original"] == diacritised_text

    def test_remove_punctuation_strips_arabic_comma(self, client: TestClient) -> None:
        text = "مرحبا، كيف حالك؟"
        response = client.post(
            "/v1/preprocess",
            json={"text": text, "remove_punctuation": True},
        )
        data = response.json()
        assert "،" not in data["processed"]
        assert "؟" not in data["processed"]

    def test_remove_numbers_strips_digits(self, client: TestClient) -> None:
        text = "عنده 42 كتاباً و١٠ أقلام"
        response = client.post(
            "/v1/preprocess",
            json={"text": text, "remove_numbers": True},
        )
        data = response.json()
        import re
        assert not re.search(r"[\d\u0660-\u0669]", data["processed"])

    def test_meta_char_count(self, client: TestClient) -> None:
        text = "مرحبا"
        response = client.post("/v1/preprocess", json={"text": text})
        data = response.json()
        assert data["meta"]["char_count"] == len(text)
