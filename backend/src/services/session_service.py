from sqlalchemy.orm import Session
from typing import Optional, List
from ..models.chat_session import ChatSession
from ..models.chat_message import ChatMessage
import logging

logger = logging.getLogger(__name__)

class SessionService:
    def __init__(self):
        pass

    def create_session(self, db: Session, user_id: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session
        """
        session = ChatSession(user_id=user_id)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    def get_session(self, db: Session, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID
        """
        try:
            import uuid
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)
            return db.query(ChatSession).filter(ChatSession.id == session_id).first()
        except ValueError:
            return None

    def get_session_with_messages(self, db: Session, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session with its messages
        """
        try:
            import uuid
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)
            return db.query(ChatSession).filter(ChatSession.id == session_id).first()
        except ValueError:
            return None

    def get_user_sessions(self, db: Session, user_id: str) -> List[ChatSession]:
        """
        Get all sessions for a specific user
        """
        return db.query(ChatSession).filter(ChatSession.user_id == user_id).all()

    def update_session(self, db: Session, session: ChatSession) -> ChatSession:
        """
        Update a chat session
        """
        db.commit()
        db.refresh(session)
        return session

    def delete_session(self, db: Session, session_id: str) -> bool:
        """
        Delete a chat session
        """
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            return True
        return False

    def get_session_messages(self, db: Session, session_id: str) -> List[ChatMessage]:
        """
        Get all messages for a specific session
        """
        try:
            import uuid
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)
            return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
        except ValueError:
            return []

# Global instance
session_service = SessionService()