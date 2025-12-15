# This model represents book content segments for the ingestion pipeline
# It's primarily used for internal tracking during ingestion,
# as the actual content is stored in Qdrant vector store
from sqlalchemy import Column, String, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class BookContent(Base):
    __tablename__ = "book_content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String, unique=True, nullable=False)  # Hash of the content to avoid duplicates
    content = Column(Text, nullable=False)  # The actual text content
    title = Column(String, nullable=True)  # Title of the section/chapter
    page_reference = Column(String, nullable=True)  # Page number or section identifier
    book_id = Column(String, nullable=True)  # Identifier for the book this content belongs to
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default={})  # Additional metadata about the content
    embedding_status = Column(String, default="pending")  # Status of embedding: pending, processed, failed

    def __repr__(self):
        return f"<BookContent(id={self.id}, title={self.title}, page_reference={self.page_reference})>"