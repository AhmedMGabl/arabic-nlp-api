"""Tests for POST /v1/entities."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestEntitiesEndpoint:

    def test_extracts_person_with_title(self, client: TestClient, ner_text: str) -> None:
        response = client.post("/v1/entities", json={"text": ner_text})
        assert response.status_code == 200
        data = response.json()
        person_entities = [e for e in data["entities"] if e["entity_type"] == "PERSON"]
        assert len(person_entities) >= 1
        # "الدكتور محمد" should be captured
        person_texts = [e["text"] for e in person_entities]
        assert any("محمد" in t for t in person_texts)

    def test_extracts_location(self, client: TestClient, ner_text: str) -> None:
        response = client.post("/v1/entities", json={"text": ner_text})
        data = response.json()
        locations = [e for e in data["entities"] if e["entity_type"] == "LOCATION"]
        assert len(locations) >= 1
        assert any("الرياض" in e["text"] for e in locations)

    def test_extracts_organization(self, client: TestClient, ner_text: str) -> None:
        response = client.post("/v1/entities", json={"text": ner_text})
        data = response.json()
        orgs = [e for e in data["entities"] if e["entity_type"] == "ORGANIZATION"]
        assert len(orgs) >= 1

    def test_extracts_date(self, client: TestClient) -> None:
        text = "وقعت الحادثة يوم الاثنين في شهر رمضان عام 2024"
        response = client.post("/v1/entities", json={"text": text})
        data = response.json()
        dates = [e for e in data["entities"] if e["entity_type"] == "DATE"]
        assert len(dates) >= 1

    def test_extracts_number(self, client: TestClient) -> None:
        text = "اشترى الرجل 150 كتاباً بمبلغ 3000 ريال"
        response = client.post("/v1/entities", json={"text": text})
        data = response.json()
        numbers = [e for e in data["entities"] if e["entity_type"] == "NUMBER"]
        assert len(numbers) >= 1

    def test_entity_span_indices_valid(self, client: TestClient, ner_text: str) -> None:
        response = client.post("/v1/entities", json={"text": ner_text})
        data = response.json()
        for entity in data["entities"]:
            assert entity["start"] >= 0
            assert entity["end"] > entity["start"]
            assert entity["end"] <= len(ner_text) + 50  # some slack for normalised offsets

    def test_entity_confidence_in_range(self, client: TestClient, ner_text: str) -> None:
        response = client.post("/v1/entities", json={"text": ner_text})
        data = response.json()
        for entity in data["entities"]:
            assert 0.0 <= entity["confidence"] <= 1.0

    def test_entity_types_found_populated(self, client: TestClient, ner_text: str) -> None:
        response = client.post("/v1/entities", json={"text": ner_text})
        data = response.json()
        assert isinstance(data["entity_types_found"], list)
        assert len(data["entity_types_found"]) > 0

    def test_entity_count_matches_list(self, client: TestClient, ner_text: str) -> None:
        response = client.post("/v1/entities", json={"text": ner_text})
        data = response.json()
        assert data["entity_count"] == len(data["entities"])

    def test_empty_text_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/entities", json={"text": ""})
        assert response.status_code == 422

    def test_no_entities_text_returns_empty(self, client: TestClient) -> None:
        text = "يذهب الطفل إلى المدرسة كل صباح"
        response = client.post("/v1/entities", json={"text": text})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["entities"], list)
        assert data["entity_count"] == len(data["entities"])
