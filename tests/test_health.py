"""Tests for GET /health."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data


def test_health_via_v1(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_returns_api_info(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "docs" in body
    assert "version" in body
