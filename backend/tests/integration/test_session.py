import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app
from src.models.chat_session import ChatSession
from src.models.chat_message import ChatMessage
from src.database import SessionLocal, engine, Base
import uuid
from datetime import datetime

client = TestClient(app)

def test_get_session_success():
    """Test retrieving an existing session"""
    # Create a mock session and message
    session_id = str(uuid.uuid4())

    with patch('src.api.v1.session.session_service') as mock_session_service:
        # Mock the session service to return a session with messages
        mock_session = MagicMock(spec=ChatSession)
        mock_session.id = session_id
        mock_session.created_at = datetime.now()
        mock_session.updated_at = datetime.now()
        mock_session.user_id = "test_user"

        mock_message = MagicMock(spec=ChatMessage)
        mock_message.id = str(uuid.uuid4())
        mock_message.session_id = session_id
        mock_message.sender = "user"
        mock_message.content = "Test message"
        mock_message.timestamp = datetime.now()
        mock_message.context = {}

        mock_session_service.get_session_with_messages.return_value = mock_session
        mock_session_service.get_session_messages.return_value = [mock_message]

        response = client.get(f"/api/v1/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "messages" in data
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Test message"

def test_get_session_not_found():
    """Test retrieving a non-existent session"""
    session_id = str(uuid.uuid4())

    with patch('src.api.v1.session.session_service') as mock_session_service:
        # Mock the session service to return None (session not found)
        mock_session_service.get_session_with_messages.return_value = None

        response = client.get(f"/api/v1/session/{session_id}")

        assert response.status_code == 404

def test_session_creation_in_query():
    """Test that sessions are properly created in the query endpoint"""
    with patch('src.services.rag_service.rag_service.query') as mock_query:
        # Mock the RAG service to return a predictable response
        mock_query.return_value = {
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
        assert "session_id" in data
        assert "message_id" in data
        assert data["answer"] == "This is a test answer"

def test_session_continuation():
    """Test that providing a session ID continues an existing session"""
    session_id = str(uuid.uuid4())

    with patch('src.services.rag_service.rag_service.query') as mock_query:
        # Mock the RAG service to return a predictable response
        mock_query.return_value = {
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

        with patch('src.services.session_service.session_service.get_session') as mock_get_session:
            # Mock the session service to return an existing session
            mock_session = MagicMock(spec=ChatSession)
            mock_session.id = session_id
            mock_get_session.return_value = mock_session

            response = client.post(
                f"/api/v1/query?session_id={session_id}",
                json={"question": "Follow-up question"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id  # Should use the same session ID