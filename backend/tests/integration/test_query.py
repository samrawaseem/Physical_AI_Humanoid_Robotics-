import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app  # Adjust import based on your main app location

client = TestClient(app)

@pytest.fixture
def mock_rag_service():
    with patch('src.api.v1.query.rag_service') as mock_service:
        yield mock_service

def test_query_endpoint_success(mock_rag_service):
    """Test successful query request"""
    # Mock the RAG service response
    mock_rag_service.query.return_value = {
        "answer": "This is a test answer",
        "sources": [
            {
                "content_snippet": "Test content snippet",
                "page_reference": "page_1",
                "similarity_score": 0.9
            }
        ],
        "retrieved_count": 1
    }

    response = client.post(
        "/api/v1/query",
        json={"question": "What is this book about?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "session_id" in data
    assert "message_id" in data
    assert data["answer"] == "This is a test answer"

def test_query_endpoint_with_selected_text(mock_rag_service):
    """Test query request with selected text context"""
    # Mock the RAG service response
    mock_rag_service.query.return_value = {
        "answer": "This is a contextual answer",
        "sources": [
            {
                "content_snippet": "Selected text context",
                "page_reference": "page_2",
                "similarity_score": 0.85
            }
        ],
        "retrieved_count": 1
    }

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What does this selected text mean?",
            "selected_text": "This is the selected text"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "This is a contextual answer"

def test_query_endpoint_empty_question():
    """Test query request with empty question"""
    response = client.post(
        "/api/v1/query",
        json={"question": ""}
    )

    assert response.status_code == 400  # Validation error

def test_query_endpoint_missing_question():
    """Test query request without question"""
    response = client.post(
        "/api/v1/query",
        json={}
    )

    assert response.status_code == 422  # Validation error