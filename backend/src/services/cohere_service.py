import cohere
from typing import List, Dict, Any
from ..config import settings
import logging

logger = logging.getLogger(__name__)

class CohereService:
    def __init__(self):
        # Initialize Cohere client
        self.client = cohere.Client(settings.cohere_api_key)
        self.model = settings.cohere_model

    def generate_response(self, prompt: str, context: List[Dict[str, str]] = None) -> str:
        """
        Generate a response using Cohere's language model
        """
        try:
            # Prepare the context for the prompt if provided
            # Cohere encourages passing documents directly to the chat endpoint for RAG
            documents = []
            if context:
                for i, ctx in enumerate(context, 1):
                    documents.append({
                        "title": f"Source {i}",
                        "snippet": ctx.get('content_snippet', '')
                    })

            # Call the Cohere API
            # We use the chat endpoint which supports RAG via 'documents'
            response = self.client.chat(
                message=prompt,
                model=self.model,
                documents=documents if documents else None,
                temperature=0.7
            )

            # Return the text response
            return response.text

        except Exception as e:
            logger.error(f"Error calling Cohere API: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text using Cohere's embedding API
        """
        try:
            response = self.client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_query" # or "search_document" depending on usage, default to query for single text
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding with Cohere: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        """
        try:
            response = self.client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type=input_type
            )
            return response.embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings with Cohere: {e}")
            raise

# Global instance
cohere_service = CohereService()
