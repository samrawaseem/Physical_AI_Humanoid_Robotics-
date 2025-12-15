import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app
from src.services.rag_service import rag_service

client = TestClient(app)

def test_query_with_page_context():
    """Test that queries can include page context for more relevant responses"""
    with patch.object(rag_service, 'query_with_full_context') as mock_query:
        # Mock the RAG service to return a predictable response
        mock_query.return_value = {
            "answer": "This is a context-aware answer based on the page content.",
            "sources": [
                {
                    "content_snippet": "Relevant content from the current page",
                    "page_reference": "current_page",
                    "similarity_score": 0.95
                }
            ],
            "retrieved_count": 1
        }

        response = client.post(
            "/api/v1/query",
            json={
                "question": "What does this page say about the topic?",
                "page_content": "This is the content of the current page that provides context for the question."
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == "This is a context-aware answer based on the page content."
        # Verify that the full context method was called
        mock_query.assert_called_once()

def test_query_with_selected_text_and_page_context():
    """Test that queries can include both selected text and page context"""
    with patch.object(rag_service, 'query_with_full_context') as mock_query:
        # Mock the RAG service to return a predictable response
        mock_query.return_value = {
            "answer": "This answer considers both the selected text and page context.",
            "sources": [
                {
                    "content_snippet": "Content related to both selected text and page",
                    "page_reference": "related_page",
                    "similarity_score": 0.88
                }
            ],
            "retrieved_count": 1
        }

        response = client.post(
            "/api/v1/query",
            json={
                "question": "Explain this concept?",
                "selected_text": "This is the text the user selected",
                "page_content": "This is the content of the current page"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "This answer considers both the selected text and page context."
        # Verify that the full context method was called with both parameters
        mock_query.assert_called_once()

def test_query_without_context_falls_back_to_general():
    """Test that queries without context still work using the general query method"""
    with patch.object(rag_service, 'query') as mock_query:
        # Mock the RAG service to return a predictable response
        mock_query.return_value = {
            "answer": "This is a general answer to your question.",
            "sources": [
                {
                    "content_snippet": "General content snippet",
                    "page_reference": "general_page",
                    "similarity_score": 0.75
                }
            ],
            "retrieved_count": 1
        }

        response = client.post(
            "/api/v1/query",
            json={
                "question": "What is this book about?"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "This is a general answer to your question."
        # Verify that the general query method was called
        mock_query.assert_called_once()

def test_context_validation():
    """Test that context parameters are properly validated"""
    # Test with very long page content (should be handled by the Pydantic model validation)
    response = client.post(
        "/api/v1/query",
        json={
            "question": "Short question",
            "page_content": "x" * 10001  # Exceeds max length of 10000
        }
    )

    # Should return a validation error
    assert response.status_code in [422]  # Validation error

def test_context_priority():
    """Test that context is properly used when available"""
    with patch.object(rag_service, 'query_with_full_context') as mock_query_with_context:
        with patch.object(rag_service, 'query') as mock_query_general:
            # Mock the context-aware RAG service to return a response
            mock_query_with_context.return_value = {
                "answer": "Context-aware answer",
                "sources": [{"content_snippet": "Context content", "page_reference": "context_page", "similarity_score": 0.9}],
                "retrieved_count": 1
            }

            response = client.post(
                "/api/v1/query",
                json={
                    "question": "Contextual question",
                    "page_content": "Page context here"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Context-aware answer"

            # Verify that the context-aware method was called instead of the general one
            mock_query_with_context.assert_called_once()
            mock_query_general.assert_not_called()