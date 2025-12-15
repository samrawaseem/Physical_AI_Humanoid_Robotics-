from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from ..config import settings
from .cohere_service import cohere_service
import logging
import uuid

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        # Initialize Qdrant client
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                port=settings.qdrant_port if settings.qdrant_port != 443 else None,
                https=True if settings.qdrant_port == 443 else False
            )
        else:
            # For local development
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )

        self.collection_name = settings.vector_collection_name
        self.embedding_size = settings.embedding_dimension

    def create_collection(self):
        """
        Create a collection in Qdrant for storing book content embeddings
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            raise

    def add_embeddings(self, texts: List[str], metadata: List[Dict[str, Any]] = None, ids: List[str] = None, embeddings: List[List[float]] = None):
        """
        Add text embeddings to the Qdrant collection
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadata is None:
            metadata = [{}] * len(texts)

        # If embeddings are not provided, we'll need to generate them separately
        # In this service, we expect the embeddings to be generated externally (e.g., by Cohere)
        if embeddings is None:
            # This would be handled by the embedding service
            raise ValueError("Embeddings must be provided separately")

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=[{
                        "content": text,
                        "metadata": meta
                    } for text, meta in zip(texts, metadata)]
                )
            )
            logger.info(f"Added {len(texts)} embeddings to Qdrant collection")
        except Exception as e:
            logger.error(f"Error adding embeddings to Qdrant: {e}")
            raise

    def add_book_content(self, content: str, metadata: Dict[str, Any] = None, content_id: str = None):
        """
        Add a single piece of book content with its embedding to Qdrant
        """
        content_id = content_id or str(uuid.uuid4())
        metadata = metadata or {}

        try:
            # Generate embedding for the content
            embedding = self.cohere_service.generate_embedding(content)

            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=[content_id],
                    vectors=[embedding],
                    payloads=[{
                        "content": content,
                        "metadata": metadata
                    }]
                )
            )
            logger.info(f"Added book content to Qdrant collection with ID: {content_id}")
            return content_id
        except Exception as e:
            logger.error(f"Error adding book content to Qdrant: {e}")
            raise

    def add_book_content_batch(self, contents: List[Dict[str, Any]]):
        """
        Add multiple pieces of book content with their embeddings to Qdrant
        Each content dict should have 'content' and 'metadata' keys
        """
        if not contents:
            return []

        ids = []
        embeddings = []
        payloads = []

        for content_dict in contents:
            content_id = str(uuid.uuid4())
            content = content_dict.get('content', '')
            metadata = content_dict.get('metadata', {})
            embedding = content_dict.get('embedding')  # Embedding should be pre-computed

            if not embedding:
                # Generate embedding if not provided
                embedding = cohere_service.generate_embedding(content)

            ids.append(content_id)
            embeddings.append(embedding)
            payloads.append({
                "content": content,
                "metadata": metadata
            })

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads
                )
            )
            logger.info(f"Added {len(contents)} book content items to Qdrant collection")
            return ids
        except Exception as e:
            logger.error(f"Error adding book content batch to Qdrant: {e}")
            raise

    def search_similar(self, query_vector: List[float], top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content in the Qdrant collection
        """
        try:
            # Prepare filters if provided
            qdrant_filters = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)
                        )
                    )
                if filter_conditions:
                    qdrant_filters = models.Filter(must=filter_conditions)

            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filters
            ).points

            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "similarity_score": hit.score
                })

            return results
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            raise

    def delete_collection(self):
        """
        Delete the collection (useful for testing/resetting)
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {e}")
            raise

# Global instance
qdrant_service = QdrantService()