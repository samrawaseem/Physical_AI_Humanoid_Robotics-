#!/usr/bin/env python3
"""
Book Content Ingestion Script

This script processes book content and ingests it into the Qdrant vector store.
It segments the text, generates embeddings, and stores them for RAG queries.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.services.text_segmentation import text_segmentation_service
from src.services.embedding_service import embedding_service
from src.services.qdrant_service import qdrant_service


def ingest_book_content(content: str, book_id: str, title: str = "Unknown Title") -> int:
    """
    Ingest a single book's content into the vector store

    Args:
        content: The full text content of the book
        book_id: Unique identifier for the book
        title: Title of the book

    Returns:
        Number of content segments processed
    """
    print(f"Starting ingestion for book: {title} (ID: {book_id})")

    # Segment the book content
    print("Segmenting content...")
    segments = text_segmentation_service.segment_text(
        content,
        metadata={
            "book_id": book_id,
            "title": title,
            "source_type": "book"
        }
    )

    print(f"Segmented into {len(segments)} chunks")

    # Generate embeddings for segments
    print("Generating embeddings...")
    embedding_records = embedding_service.generate_embeddings_for_segments(segments)

    # Filter out any segments that failed to generate embeddings
    valid_embeddings = [rec for rec in embedding_records if rec["embedding"] is not None]
    if len(valid_embeddings) != len(embedding_records):
        print(f"Warning: {len(embedding_records) - len(valid_embeddings)} segments failed to generate embeddings")

    print(f"Generated embeddings for {len(valid_embeddings)} segments")

    # Prepare content for Qdrant
    qdrant_contents = []
    for record in valid_embeddings:
        qdrant_contents.append({
            "content": record["content"],
            "metadata": record["metadata"],
            "embedding": record["embedding"]
        })

    # Add to Qdrant vector store
    print("Adding to Qdrant vector store...")
    content_ids = qdrant_service.add_book_content_batch(qdrant_contents)

    print(f"Successfully added {len(content_ids)} content segments to Qdrant")
    print(f"Ingestion completed for book: {title}")

    return len(content_ids)


def ingest_from_file(file_path: str, book_id: str, title: str = None) -> int:
    """
    Ingest book content from a text file

    Args:
        file_path: Path to the text file containing book content
        book_id: Unique identifier for the book
        title: Title of the book (defaults to filename if not provided)

    Returns:
        Number of content segments processed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use filename as title if not provided
    if title is None:
        title = Path(file_path).stem

    print(f"Reading content from: {file_path}")

    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return ingest_book_content(content, book_id, title)


def main():
    parser = argparse.ArgumentParser(description="Ingest book content into RAG system")
    parser.add_argument("source", help="Path to the book content file or directory")
    parser.add_argument("--book-id", required=True, help="Unique identifier for the book")
    parser.add_argument("--title", help="Title of the book (optional, defaults to filename)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size for segmentation (default: 1000)")

    args = parser.parse_args()

    # Validate configuration
    if not settings.qdrant_url:
        print("Error: QDRANT_URL not configured in environment")
        sys.exit(1)

    if not settings.cohere_api_key:
        print("Error: COHERE_API_KEY not configured in environment")
        sys.exit(1)

    # Set chunk size if provided
    if args.chunk_size != 1000:
        from src.services.text_segmentation import text_segmentation_service
        text_segmentation_service.max_chunk_size = args.chunk_size

    try:
        # Ensure collection exists
        print("Ensuring Qdrant collection exists...")
        qdrant_service.create_collection()

        # Check if source is a file or directory
        source_path = Path(args.source)

        if source_path.is_file():
            # Process single file
            processed_count = ingest_from_file(
                file_path=str(source_path),
                book_id=args.book_id,
                title=args.title
            )
        elif source_path.is_dir():
            # Process all text files in directory (recursively)
            text_files = list(source_path.rglob("*.txt")) + list(source_path.rglob("*.md"))
            if not text_files:
                print(f"No text files found in directory: {args.source}")
                sys.exit(1)

            total_processed = 0
            for text_file in text_files:
                print(f"\nProcessing file: {text_file.name}")
                processed_count = ingest_from_file(
                    file_path=str(text_file),
                    book_id=f"{args.book_id}_{text_file.stem}",
                    title=f"{args.title or args.book_id} - {text_file.stem}"
                )
                total_processed += processed_count

            processed_count = total_processed
        else:
            print(f"Error: Source not found: {args.source}")
            sys.exit(1)

        print(f"\nIngestion complete! Processed {processed_count} content segments.")

    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()