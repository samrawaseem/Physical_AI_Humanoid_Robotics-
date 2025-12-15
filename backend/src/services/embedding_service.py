from typing import List, Dict, Any
from .cohere_service import cohere_service
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.cohere_service = cohere_service

    def generate_embeddings_for_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text segments
        """
        embeddings = []

        for segment in segments:
            try:
                # Generate embedding for the content
                embedding_vector = self.cohere_service.generate_embedding(segment["content"])

                # Create the embedding record
                embedding_record = {
                    "content": segment["content"],
                    "embedding": embedding_vector,
                    "metadata": segment.get("metadata", {})
                }

                embeddings.append(embedding_record)

            except Exception as e:
                logger.error(f"Error generating embedding for segment: {e}")
                # Add the segment without embedding so the process continues
                embedding_record = {
                    "content": segment["content"],
                    "embedding": None,
                    "metadata": {**segment.get("metadata", {}), "error": str(e)}
                }
                embeddings.append(embedding_record)

        logger.info(f"Generated embeddings for {len(embeddings)} segments")
        return embeddings

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding for the given text
        """
        return self.cohere_service.generate_embedding(text)

    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        """
        return self.cohere_service.generate_embeddings_batch(texts)

# Global instance
embedding_service = EmbeddingService()