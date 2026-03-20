"""Tests for POST /v1/detect-dialect."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestDialectEndpoint:

    def test_egyptian_detected(self, client: TestClient, egyptian_text: str) -> None:
        response = client.post("/v1/detect-dialect", json={"text": egyptian_text})
        assert response.status_code == 200
        data = response.json()
        assert data["dialect"] == "EGY"
        assert data["confidence"] > 0.3

    def test_gulf_detected(self, client: TestClient, gulf_text: str) -> None:
        response = client.post("/v1/detect-dialect", json={"text": gulf_text})
        assert response.status_code == 200
        data = response.json()
        assert data["dialect"] == "GULF"

    def test_levantine_detected(self, client: TestClient, levantine_text: str) -> None:
        response = client.post("/v1/detect-dialect", json={"text": levantine_text})
        assert response.status_code == 200
        data = response.json()
        assert data["dialect"] == "LEV"

    def test_response_has_all_five_dialects(self, client: TestClient, egyptian_text: str) -> None:
        response = client.post("/v1/detect-dialect", json={"text": egyptian_text})
        data = response.json()
        codes = {d["code"] for d in data["all_scores"]}
        assert codes == {"MSA", "EGY", "GULF", "LEV", "MAG"}

    def test_all_scores_sum_to_approximately_one(self, client: TestClient, egyptian_text: str) -> None:
        response = client.post("/v1/detect-dialect", json={"text": egyptian_text})
        data = response.json()
        total = sum(d["score"] for d in data["all_scores"])
        assert abs(total - 1.0) < 0.02

    def test_dialect_names_present(self, client: TestClient, gulf_text: str) -> None:
        response = client.post("/v1/detect-dialect", json={"text": gulf_text})
        data = response.json()
        assert data["dialect_name_en"]
        assert data["dialect_name_ar"]

    def test_meta_present(self, client: TestClient, egyptian_text: str) -> None:
        response = client.post("/v1/detect-dialect", json={"text": egyptian_text})
        data = response.json()
        assert data["meta"]["char_count"] == len(egyptian_text)
        assert data["meta"]["processing_time_ms"] >= 0

    def test_empty_text_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/detect-dialect", json={"text": ""})
        assert response.status_code == 422

    def test_confidence_in_valid_range(self, client: TestClient, gulf_text: str) -> None:
        response = client.post("/v1/detect-dialect", json={"text": gulf_text})
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0
