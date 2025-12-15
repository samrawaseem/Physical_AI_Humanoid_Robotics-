from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Text, Uuid
# from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..database import Base

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(Uuid(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    sender = Column(String, nullable=False)  # "user" or "bot"
    content = Column(Text, nullable=False)  # The actual message content
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    context = Column(JSON, default={})  # Additional context (e.g., selected text)

    # Relationship
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, sender={self.sender})>"

# Add relationship to ChatSession
# Relationship will be set up in chat_session.py to avoid circular import