import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app
from src.services.rag_service import rag_service

client = TestClient(app)

def test_basic_question_answering():
    """
    Test basic question answering functionality
    This test verifies that the system can process a simple question and return an answer
    """
    with patch.object(rag_service, 'query') as mock_query:
        # Mock the RAG service to return a predictable response
        mock_query.return_value = {
            "answer": "This is a test answer for your question.",
            "sources": [
                {
                    "content_snippet": "Relevant content snippet",
                    "page_reference": "chapter_1",
                    "similarity_score": 0.85
                }
            ],
            "retrieved_count": 1
        }

        # Send a test question
        response = client.post(
            "/api/v1/query",
            json={
                "question": "What is this book about?"
            }
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert "message_id" in data

        assert data["answer"] == "This is a test answer for your question."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["content_snippet"] == "Relevant content snippet"

def test_question_answering_with_context():
    """
    Test question answering with selected text context
    """
    with patch.object(rag_service, 'query') as mock_query:
        # Mock the RAG service to return a predictable response
        mock_query.return_value = {
            "answer": "Based on the selected text, the answer is...",
            "sources": [
                {
                    "content_snippet": "Selected text content",
                    "page_reference": "page_5",
                    "similarity_score": 0.92
                }
            ],
            "retrieved_count": 1
        }

        # Send a test question with selected text
        response = client.post(
            "/api/v1/query",
            json={
                "question": "What does this selected text mean?",
                "selected_text": "This is the text that was selected by the user"
            }
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert data["answer"] == "Based on the selected text, the answer is..."

def test_health_endpoint():
    """
    Test the health check endpoint
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"