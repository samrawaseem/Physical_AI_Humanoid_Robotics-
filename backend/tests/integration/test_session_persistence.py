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

def test_session_persistence_across_requests():
    """
    Test that session data persists across multiple requests
    """
    # Mock RAG service responses
    first_response = {
        "answer": "First answer to the question",
        "sources": [
            {
                "content_snippet": "First content snippet",
                "page_reference": "page_1",
                "similarity_score": 0.9
            }
        ],
        "retrieved_count": 1
    }

    second_response = {
        "answer": "Second answer to the follow-up question",
        "sources": [
            {
                "content_snippet": "Second content snippet",
                "page_reference": "page_2",
                "similarity_score": 0.85
            }
        ],
        "retrieved_count": 1
    }

    session_id = str(uuid.uuid4())

    with patch('src.services.rag_service.rag_service.query') as mock_query:
        with patch('src.services.session_service.session_service.get_session') as mock_get_session:
            with patch('src.services.session_service.session_service.create_session') as mock_create_session:
                # Mock the session service to return an existing session
                mock_session = MagicMock(spec=ChatSession)
                mock_session.id = session_id
                mock_get_session.return_value = mock_session
                mock_create_session.return_value = mock_session

                # First query - create session
                mock_query.return_value = first_response
                response1 = client.post(
                    "/api/v1/query",
                    json={"question": "What is this book about?"}
                )

                assert response1.status_code == 200
                data1 = response1.json()
                first_session_id = data1["session_id"]
                first_message_id = data1["message_id"]

                # Second query - use same session
                mock_query.return_value = second_response
                response2 = client.post(
                    f"/api/v1/query?session_id={first_session_id}",
                    json={"question": "Can you elaborate?"}
                )

                assert response2.status_code == 200
                data2 = response2.json()
                assert data2["session_id"] == first_session_id  # Same session
                assert data2["answer"] == "Second answer to the follow-up question"

def test_session_with_messages_retrieval():
    """
    Test that we can retrieve a session with its messages
    """
    session_id = str(uuid.uuid4())

    # Create mock session and messages
    mock_session = MagicMock(spec=ChatSession)
    mock_session.id = session_id
    mock_session.created_at = datetime.now()
    mock_session.updated_at = datetime.now()
    mock_session.user_id = "test_user"

    mock_message1 = MagicMock(spec=ChatMessage)
    mock_message1.id = str(uuid.uuid4())
    mock_message1.session_id = session_id
    mock_message1.sender = "user"
    mock_message1.content = "First question"
    mock_message1.timestamp = datetime.now()
    mock_message1.context = {}

    mock_message2 = MagicMock(spec=ChatMessage)
    mock_message2.id = str(uuid.uuid4())
    mock_message2.session_id = session_id
    mock_message2.sender = "bot"
    mock_message2.content = "First answer"
    mock_message2.timestamp = datetime.now()
    mock_message2.context = {"sources": []}

    with patch('src.api.v1.session.session_service') as mock_session_service:
        mock_session_service.get_session_with_messages.return_value = mock_session
        mock_session_service.get_session_messages.return_value = [mock_message1, mock_message2]

        response = client.get(f"/api/v1/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "First question"
        assert data["messages"][1]["content"] == "First answer"

def test_new_session_creation():
    """
    Test that a new session is created when no session_id is provided
    """
    with patch('src.services.rag_service.rag_service.query') as mock_query:
        with patch('src.services.session_service.session_service.create_session') as mock_create_session:
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

            # Mock the session creation
            new_session = MagicMock(spec=ChatSession)
            new_session.id = str(uuid.uuid4())
            mock_create_session.return_value = new_session

            response = client.post(
                "/api/v1/query",
                json={"question": "What is this book about?"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            # Verify that create_session was called (no session_id passed in URL)
            assert data["session_id"] == str(new_session.id)