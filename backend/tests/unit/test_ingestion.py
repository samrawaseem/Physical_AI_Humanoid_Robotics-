import pytest
from unittest.mock import patch, MagicMock
from src.services.text_segmentation import text_segmentation_service
from src.services.embedding_service import embedding_service
from src.services.qdrant_service import qdrant_service
from src.models.book_content import BookContent
from src.scripts.ingest_books import ingest_book_content


def test_text_segmentation_basic():
    """Test basic text segmentation functionality"""
    text = "This is a sentence. This is another sentence! Is this a question?"

    segments = text_segmentation_service.segment_text(text, metadata={"test": "value"})

    assert len(segments) > 0
    assert all("content" in segment for segment in segments)
    assert all("metadata" in segment for segment in segments)
    assert segments[0]["metadata"]["test"] == "value"


def test_text_segmentation_long_text():
    """Test segmentation of longer text"""
    long_text = "First sentence. " * 50  # Create a longer text

    # Temporarily set smaller chunk size for testing
    original_size = text_segmentation_service.max_chunk_size
    text_segmentation_service.max_chunk_size = 50

    try:
        segments = text_segmentation_service.segment_text(long_text)

        assert len(segments) > 1  # Should be split into multiple segments
        assert all(len(segment["content"]) <= original_size for segment in segments)
    finally:
        # Restore original size
        text_segmentation_service.max_chunk_size = original_size


def test_text_segmentation_empty_text():
    """Test segmentation of empty text"""
    segments = text_segmentation_service.segment_text("", metadata={"empty": "test"})

    assert len(segments) == 0


def test_text_segmentation_with_metadata():
    """Test that metadata is properly preserved in segments"""
    text = "Test sentence one. Test sentence two."
    metadata = {"author": "test_author", "year": "2023"}

    segments = text_segmentation_service.segment_text(text, metadata=metadata)

    assert len(segments) > 0
    for segment in segments:
        assert segment["metadata"]["author"] == "test_author"
        assert segment["metadata"]["year"] == "2023"


def test_embedding_generation():
    """Test embedding generation for segments"""
    with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
        # Mock embedding generation
        mock_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # Simplified embedding

        segments = [
            {"content": "Test content 1", "metadata": {"id": 1}},
            {"content": "Test content 2", "metadata": {"id": 2}}
        ]

        embeddings = embedding_service.generate_embeddings_for_segments(segments)

        assert len(embeddings) == 2
        assert embeddings[0]["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert embeddings[0]["content"] == "Test content 1"
        assert embeddings[1]["metadata"]["id"] == 2


def test_embedding_generation_with_error():
    """Test embedding generation handles errors gracefully"""
    with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
        # Mock first call to succeed, second to fail
        mock_embedding.side_effect = [Exception("API Error"), [0.5, 0.4, 0.3, 0.2, 0.1]]

        segments = [
            {"content": "Test content 1", "metadata": {"id": 1}},
            {"content": "Test content 2", "metadata": {"id": 2}}
        ]

        embeddings = embedding_service.generate_embeddings_for_segments(segments)

        assert len(embeddings) == 2
        # First should have error metadata
        assert "error" in embeddings[0]["metadata"]
        # Second should have proper embedding
        assert embeddings[1]["embedding"] == [0.5, 0.4, 0.3, 0.2, 0.1]


def test_qdrant_batch_add():
    """Test adding multiple content items to Qdrant"""
    with patch.object(qdrant_service.client, 'upsert') as mock_upsert:
        mock_upsert.return_value = MagicMock()

        contents = [
            {
                "content": "Test content 1",
                "metadata": {"id": 1},
                "embedding": [0.1, 0.2, 0.3]
            },
            {
                "content": "Test content 2",
                "metadata": {"id": 2},
                "embedding": [0.4, 0.5, 0.6]
            }
        ]

        content_ids = qdrant_service.add_book_content_batch(contents)

        assert len(content_ids) == 2
        # Verify upsert was called
        mock_upsert.assert_called_once()


def test_ingest_book_content_function():
    """Test the main ingestion function"""
    book_content = "This is a test book. It has multiple sentences. Some are short. Others are longer and contain more detail."
    book_id = "test-book-123"
    title = "Test Book"

    with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
        with patch.object(qdrant_service, 'add_book_content_batch') as mock_add_batch:
            # Mock embedding generation
            mock_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            # Mock Qdrant batch add
            mock_add_batch.return_value = ["id1", "id2", "id3"]

            result_count = ingest_book_content(book_content, book_id, title)

            # Should return the number of segments processed
            assert result_count > 0
            # Verify the segmentation happened
            mock_embedding.assert_called()  # Should be called for each segment
            # Verify the content was added to Qdrant
            mock_add_batch.assert_called_once()


def test_book_content_model():
    """Test the BookContent model structure"""
    from datetime import datetime

    content = BookContent(
        content="Test book content",
        title="Test Title",
        page_reference="page 1",
        book_id="test-book"
    )

    assert content.content == "Test book content"
    assert content.title == "Test Title"
    assert content.page_reference == "page 1"
    assert content.book_id == "test-book"
    assert content.embedding_status == "pending"
    assert isinstance(content.created_at, datetime)
    assert isinstance(content.updated_at, datetime)


def test_text_segmentation_with_special_characters():
    """Test that text segmentation handles special characters properly"""
    text = "This is a test with \"quotes\", (parentheses), and other punctuation! What about newlines?\nAnd tabs?\tTab content."

    segments = text_segmentation_service.segment_text(text)

    assert len(segments) > 0
    # Should not contain special formatting characters that were cleaned
    combined_content = " ".join([seg["content"] for seg in segments])
    # Should still have sentence structure
    assert "." in combined_content or "!" in combined_content or "?" in combined_content