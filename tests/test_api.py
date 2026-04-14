"""
tests/test_api.py
-----------------
Integration tests for the RAG Chatbot API.
Run: uv run pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.core.dependencies import get_rag_chain, get_vector_store, get_ingestion_service
from app.models.schemas import ChatResponse, Source, CollectionInfo

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_rag_chain():
    mock = MagicMock()
    mock.query.return_value = ChatResponse(
        answer="Đây là câu trả lời mẫu.",
        sources=[Source(content="Nội dung mẫu", source="test.txt", score=0.85)],
        model="llama3.2:3b",
        latency_ms=120.5,
    )
    return mock


@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.get_collection_info.return_value = {
        "name": "test_collection",
        "document_count": 42,
        "embedding_model": "nomic-embed-text",
    }
    return mock


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


# ── Chat ──────────────────────────────────────────────────────────────────────

def test_chat_query_success(mock_rag_chain):
    app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain
    response = client.post("/chat/query", json={"question": "AI là gì?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "model" in data
    assert "latency_ms" in data
    app.dependency_overrides.clear()


def test_chat_query_empty_question():
    response = client.post("/chat/query", json={"question": ""})
    assert response.status_code == 422


def test_chat_query_too_long():
    response = client.post("/chat/query", json={"question": "a" * 2001})
    assert response.status_code == 422


def test_collection_info(mock_vector_store):
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    response = client.get("/chat/collection")
    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] == 42
    app.dependency_overrides.clear()


# ── Ingestion ─────────────────────────────────────────────────────────────────

def test_ingest_unsupported_file_type():
    response = client.post(
        "/ingest/file",
        files={"file": ("test.exe", b"binary content", "application/octet-stream")},
    )
    assert response.status_code == 422


def test_ingest_valid_txt_file():
    mock_service = MagicMock()
    mock_service.ingest_file.return_value = 5
    app.dependency_overrides[get_ingestion_service] = lambda: mock_service

    response = client.post(
        "/ingest/file",
        files={"file": ("sample.txt", b"Hello world. This is test content.", "text/plain")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["documents_added"] == 5
    app.dependency_overrides.clear()


# ── Evaluation ────────────────────────────────────────────────────────────────

def test_eval_empty_samples():
    response = client.post("/eval/run", json={"samples": []})
    assert response.status_code == 422
