from sqlalchemy import Column, String, DateTime, JSON, Uuid
# from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..database import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    user_id = Column(String, nullable=True)  # Optional identifier for authenticated users
    session_metadata = Column(JSON, default={})  # Additional session-specific data

    # Relationship - ChatMessage relationship defined here to avoid circular import
    messages = relationship("ChatMessage", order_by="ChatMessage.timestamp", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, created_at={self.created_at})>"