import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app
from src.services.rag_service import rag_service

client = TestClient(app)

def test_context_aware_response_functionality():
    """
    Test the overall functionality of context-aware responses
    """
    with patch.object(rag_service, 'query_with_full_context') as mock_query_with_context:
        # Mock response when context is provided
        context_response = {
            "answer": "Based on the current page content about neural networks, this concept refers to...",
            "sources": [
                {
                    "content_snippet": "Neural networks are computing systems inspired by the human brain...",
                    "page_reference": "chapter_5",
                    "similarity_score": 0.92
                }
            ],
            "retrieved_count": 1
        }
        mock_query_with_context.return_value = context_response

        # Test 1: Query with page content should use context
        response = client.post(
            "/api/v1/query",
            json={
                "question": "What are neural networks?",
                "page_content": "Neural networks are computing systems inspired by the human brain. They consist of layers of interconnected nodes that process information in a way similar to neurons in the brain."
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "neural networks" in data["answer"].lower()
        assert len(data["sources"]) > 0

        # Verify the context-aware method was called
        mock_query_with_context.assert_called_once()

        # Reset the mock for the next test
        mock_query_with_context.reset_mock()

        # Test 2: Query with both selected text and page content
        mock_query_with_context.return_value = {
            "answer": "The selected text explains how neural networks function within the broader context of the chapter.",
            "sources": [
                {
                    "content_snippet": "The activation function determines the output of a neural network node...",
                    "page_reference": "section_5.2",
                    "similarity_score": 0.89
                }
            ],
            "retrieved_count": 1
        }

        response2 = client.post(
            "/api/v1/query",
            json={
                "question": "How does this work?",
                "selected_text": "The activation function determines the output of a neural network node",
                "page_content": "Chapter 5 discusses neural networks in detail. Neural networks are computing systems..."
            }
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert "answer" in data2
        assert len(data2["sources"]) > 0

        # Verify the context-aware method was called again
        mock_query_with_context.assert_called_once()

def test_context_vs_general_query():
    """
    Test that context-aware queries produce different results than general queries
    """
    with patch.object(rag_service, 'query_with_full_context') as mock_context_query:
        with patch.object(rag_service, 'query') as mock_general_query:
            # Mock different responses for context vs general
            context_response = {
                "answer": "In the context of this chapter, neural networks refer to the specific implementation described here...",
                "sources": [{"content_snippet": "Specific implementation details", "page_reference": "current_page", "similarity_score": 0.95}],
                "retrieved_count": 1
            }
            general_response = {
                "answer": "Neural networks are a general concept in machine learning involving interconnected nodes...",
                "sources": [{"content_snippet": "General definition", "page_reference": "intro_chapter", "similarity_score": 0.80}],
                "retrieved_count": 1
            }

            mock_context_query.return_value = context_response
            mock_general_query.return_value = general_response

            # Query with context
            response_with_context = client.post(
                "/api/v1/query",
                json={
                    "question": "What are neural networks?",
                    "page_content": "This chapter describes a specific neural network architecture"
                }
            )

            # Query without context
            response_without_context = client.post(
                "/api/v1/query",
                json={
                    "question": "What are neural networks?"
                }
            )

            assert response_with_context.status_code == 200
            assert response_without_context.status_code == 200

            data_with_context = response_with_context.json()
            data_without_context = response_without_context.json()

            # Verify different methods were called
            mock_context_query.assert_called_once()
            mock_general_query.assert_called_once()

            # The responses should be different due to context
            assert data_with_context["answer"] != data_without_context["answer"]

def test_empty_context_handling():
    """
    Test that empty or whitespace-only context is handled properly
    """
    with patch.object(rag_service, 'query') as mock_general_query:
        # Mock a general response
        mock_general_query.return_value = {
            "answer": "General answer when no meaningful context provided",
            "sources": [{"content_snippet": "General info", "page_reference": "general", "similarity_score": 0.7}],
            "retrieved_count": 1
        }

        # Test with empty page content
        response = client.post(
            "/api/v1/query",
            json={
                "question": "What is this about?",
                "page_content": ""  # Empty context
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

        # Should fall back to general query when context is empty
        mock_general_query.assert_called_once()

def test_context_length_limits():
    """
    Test that very long context content is handled appropriately
    """
    with patch.object(rag_service, 'query_with_full_context') as mock_query:
        mock_query.return_value = {
            "answer": "Answer based on lengthy context",
            "sources": [{"content_snippet": "Relevant snippet", "page_reference": "some_page", "similarity_score": 0.85}],
            "retrieved_count": 1
        }

        # Test with long but acceptable page content
        long_content = "This is a lengthy page content. " * 400  # About 16000 characters, under the 10000 limit

        response = client.post(
            "/api/v1/query",
            json={
                "question": "Summarize this content",
                "page_content": long_content
            }
        )

        # This should fail validation since content exceeds 10000 chars
        assert response.status_code == 422  # Validation error

    # Test with content within limits
    with patch.object(rag_service, 'query_with_full_context') as mock_query:
        mock_query.return_value = {
            "answer": "Answer based on long but acceptable context",
            "sources": [{"content_snippet": "Relevant snippet", "page_reference": "some_page", "similarity_score": 0.85}],
            "retrieved_count": 1
        }

        acceptable_content = "This is a lengthy page content. " * 150  # About 6000 characters, under the 10000 limit

        response = client.post(
            "/api/v1/query",
            json={
                "question": "Summarize this content",
                "page_content": acceptable_content
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data