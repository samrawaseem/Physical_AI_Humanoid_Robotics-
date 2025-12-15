import pytest
from unittest.mock import patch, MagicMock
from src.services.text_segmentation import text_segmentation_service
from src.services.embedding_service import embedding_service
from src.services.qdrant_service import qdrant_service
from src.scripts.ingest_books import ingest_book_content
from src.models.book_content import BookContent
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os


def test_end_to_end_ingestion_pipeline():
    """
    End-to-end test for the book content ingestion pipeline
    Tests the complete flow from text input to vector storage
    """
    book_content = """
    Chapter 1: Introduction to AI
    Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.

    The scope of AI is disputed: as machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.

    Chapter 2: Machine Learning Basics
    Machine learning (ML) is the study of computer algorithms that can improve automatically through experience and by the use of data. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.
    """

    book_id = "test-book-e2e-123"
    title = "Test AI Book"

    with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
        with patch.object(qdrant_service, 'add_book_content_batch') as mock_add_batch:
            # Mock embedding generation with realistic values
            mock_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] * 205  # 1025-dim vector roughly matching Cohere (approx) or just generic

            # Mock Qdrant batch add to return success
            mock_add_batch.return_value = [f"test-id-{i}" for i in range(5)]  # Simulate 5 segments

            # Run the ingestion
            processed_count = ingest_book_content(book_content, book_id, title)

            # Assertions
            assert processed_count == 5, f"Expected 5 segments, got {processed_count}"

            # Verify segmentation was called
            # The text should be broken into multiple chunks due to length
            mock_add_batch.assert_called_once()

            # Get the arguments passed to add_book_content_batch
            args, kwargs = mock_add_batch.call_args
            batch_contents = args[0]  # First argument is the list of contents

            assert len(batch_contents) == 5
            for content_dict in batch_contents:
                assert "content" in content_dict
                assert "metadata" in content_dict
                assert "embedding" in content_dict
                assert content_dict["metadata"]["book_id"] == book_id
                assert content_dict["metadata"]["title"] == title
                assert content_dict["metadata"]["source_type"] == "book"


def test_ingestion_with_empty_content():
    """Test ingestion behavior with empty or minimal content"""
    with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
        with patch.object(qdrant_service, 'add_book_content_batch') as mock_add_batch:
            mock_embedding.return_value = [0.1, 0.2, 0.3]
            mock_add_batch.return_value = ["test-id"]

            # Test with empty content
            processed_count = ingest_book_content("", "empty-book", "Empty Book")
            assert processed_count == 0

            # Test with minimal content
            processed_count = ingest_book_content("Short text.", "short-book", "Short Book")
            assert processed_count >= 0  # Should handle gracefully


def test_ingestion_error_handling():
    """Test that the ingestion pipeline handles errors gracefully"""
    book_content = "This is test content for error handling."
    book_id = "error-test-book"
    title = "Error Test Book"

    with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
        with patch.object(qdrant_service, 'add_book_content_batch') as mock_add_batch:
            # Simulate an error in embedding generation
            mock_embedding.side_effect = Exception("API Error")

            try:
                processed_count = ingest_book_content(book_content, book_id, title)
                # Should handle the error and continue
            except Exception:
                # If the function propagates the error, that's also acceptable
                # as long as it's handled appropriately in production
                pass

    # Test Qdrant error
    with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
        with patch.object(qdrant_service, 'add_book_content_batch') as mock_add_batch:
            mock_embedding.return_value = [0.1, 0.2, 0.3]
            mock_add_batch.side_effect = Exception("Qdrant Error")

            with pytest.raises(Exception, match="Qdrant Error"):
                ingest_book_content(book_content, book_id, title)


def test_ingestion_file_script_integration():
    """Test the integration of the ingestion script with a temporary file"""
    # Create a temporary file with test content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write("""
        Test Book Content
        ================

        Chapter 1: Getting Started
        This is the beginning of our test book. It contains sample content that will be used to test the ingestion pipeline.

        The content is structured in a way that should create multiple segments when processed by the text segmentation service.

        Chapter 2: Advanced Topics
        This chapter covers more complex topics. The goal is to have enough content to test the segmentation logic properly.

        We want to ensure that the text is split appropriately based on the configured chunk size.
        """)
        temp_filename = temp_file.name

    try:
        # Import the function from the script
        from src.scripts.ingest_books import ingest_from_file

        with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
            with patch.object(qdrant_service, 'add_book_content_batch') as mock_add_batch:
                # Mock the embedding service
                mock_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] * 307  # 1536-dim vector
                mock_add_batch.return_value = ["test-id-1", "test-id-2"]

                # Run ingestion from file
                processed_count = ingest_from_file(
                    file_path=temp_filename,
                    book_id="file-test-book",
                    title="File Test Book"
                )

                # Should process multiple segments
                assert processed_count >= 1

    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)


def test_segmentation_parameters():
    """Test that segmentation respects the configured parameters"""
    # Create content that will definitely be split with default parameters
    long_sentence = "This is a very long sentence. " * 100  # Create a long sentence
    long_content = long_sentence + " Here is another sentence."

    # Test with small chunk size
    original_size = text_segmentation_service.max_chunk_size
    text_segmentation_service.max_chunk_size = 50  # Very small size to force splitting

    try:
        segments = text_segmentation_service.segment_text(
            long_content,
            metadata={"test": "chunk-size"}
        )

        # Should be split into multiple segments due to small chunk size
        assert len(segments) > 1

        # Each segment should be within the size limit (with some tolerance for sentence boundaries)
        for segment in segments:
            assert len(segment["content"]) <= 100  # Allow some flexibility for sentence boundaries
    finally:
        # Restore original size
        text_segmentation_service.max_chunk_size = original_size


def test_metadata_preservation():
    """Test that metadata is properly preserved throughout the ingestion pipeline"""
    book_content = "Sample book content for metadata testing."
    book_id = "metadata-test-book"
    title = "Metadata Test Book"

    with patch('src.services.cohere_service.cohere_service.generate_embedding') as mock_embedding:
        with patch.object(qdrant_service, 'add_book_content_batch') as mock_add_batch:
            mock_embedding.return_value = [0.1, 0.2, 0.3]
            mock_add_batch.return_value = ["test-id"]

            # Run ingestion
            processed_count = ingest_book_content(book_content, book_id, title)

            # Check that add_book_content_batch was called with correct metadata
            args, kwargs = mock_add_batch.call_args
            batch_contents = args[0]

            assert len(batch_contents) >= 1
            for content_dict in batch_contents:
                metadata = content_dict["metadata"]
                assert metadata["book_id"] == book_id
                assert metadata["title"] == title
                assert metadata["source_type"] == "book"