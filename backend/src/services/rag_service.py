from typing import List, Dict, Any, Optional
from .qdrant_service import qdrant_service
from .cohere_service import cohere_service
from ..config import settings
from ..api.errors import VectorStoreError, LLMError
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.qdrant_service = qdrant_service
        self.cohere_service = cohere_service

    def query(self, question: str, selected_text: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Main RAG query function that retrieves relevant documents and generates an answer
        """
        try:
            # Generate embedding for the question
            question_embedding = self.cohere_service.generate_embedding(question)

            # Prepare filters based on selected text context
            filters = {}
            if selected_text:
                # If selected text is provided, we might want to search within that context
                # For now, we'll just include the selected text in our context
                filters = {}  # We can add more specific filters later if needed

            # Search for relevant documents in Qdrant
            search_results = self.qdrant_service.search_similar(
                query_vector=question_embedding,
                top_k=top_k,
                filters=filters
            )

            # Prepare context for the LLM from the search results
            context = []
            for result in search_results:
                context.append({
                    "content_snippet": result["content"],
                    "page_reference": result["metadata"].get("page_reference", ""),
                    "similarity_score": result["similarity_score"]
                })

            # Generate the answer using Cohere
            answer = self.cohere_service.generate_response(
                prompt=question,
                context=context
            )

            # Return the answer with sources
            return {
                "answer": answer,
                "sources": context,
                "retrieved_count": len(search_results)
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise

    def query_with_page_context(self, question: str, page_content: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query function that considers the current page context
        """
        try:
            # Generate embedding for the question
            question_embedding = self.cohere_service.generate_embedding(question)

            # Create a combined query that considers both the question and page context
            combined_query = f"{question} Context: {page_content}" if page_content else question
            combined_embedding = self.cohere_service.generate_embedding(combined_query)

            # Search for relevant documents in Qdrant
            search_results = self.qdrant_service.search_similar(
                query_vector=combined_embedding,
                top_k=top_k
            )

            # Prepare context for the LLM from the search results
            context = []
            for result in search_results:
                context.append({
                    "content_snippet": result["content"],
                    "page_reference": result["metadata"].get("page_reference", ""),
                    "similarity_score": result["similarity_score"]
                })

            # Generate the answer using Cohere, including page content as additional context
            answer = self.cohere_service.generate_response(
                prompt=question,
                context=context
            )

            # Return the answer with sources
            return {
                "answer": answer,
                "sources": context,
                "retrieved_count": len(search_results)
            }

        except Exception as e:
            logger.error(f"Error in RAG query with page context: {e}")
            raise

    def query_with_full_context(self, question: str, selected_text: Optional[str] = None, page_content: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Enhanced query function that can consider both selected text and page content
        """
        try:
            # Generate embedding for the question
            question_embedding = self.cohere_service.generate_embedding(question)

            # Create a combined query based on available context
            context_parts = []
            if selected_text:
                context_parts.append(f"Selected text: {selected_text}")
            if page_content:
                context_parts.append(f"Current page content: {page_content}")

            if context_parts:
                combined_query = f"{question} {'; '.join(context_parts)}"
            else:
                combined_query = question

            combined_embedding = self.cohere_service.generate_embedding(combined_query)

            # Search for relevant documents in Qdrant
            search_results = self.qdrant_service.search_similar(
                query_vector=combined_embedding,
                top_k=top_k
            )

            # Prepare context for the LLM from the search results
            context = []
            for result in search_results:
                context.append({
                    "content_snippet": result["content"],
                    "page_reference": result["metadata"].get("page_reference", ""),
                    "similarity_score": result["similarity_score"]
                })

            # Generate the answer using Cohere
            answer = self.cohere_service.generate_response(
                prompt=question,
                context=context
            )

            # Return the answer with sources
            return {
                "answer": answer,
                "sources": context,
                "retrieved_count": len(search_results)
            }

        except Exception as e:
            logger.error(f"Error in RAG query with full context: {e}")
            raise

    def robust_query(self, question: str, selected_text: Optional[str] = None, page_content: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Robust query function with comprehensive error handling
        """
        try:
            # Generate embedding for the question
            question_embedding = self.cohere_service.generate_embedding(question)

            # Create a combined query based on available context
            context_parts = []
            if selected_text:
                context_parts.append(f"Selected text: {selected_text}")
            if page_content:
                context_parts.append(f"Current page content: {page_content}")

            if context_parts:
                combined_query = f"{question} {'; '.join(context_parts)}"
            else:
                combined_query = question

            combined_embedding = self.cohere_service.generate_embedding(combined_query)

            # Search for relevant documents in Qdrant with error handling
            try:
                search_results = self.qdrant_service.search_similar(
                    query_vector=combined_embedding,
                    top_k=top_k
                )
            except Exception as vector_store_error:
                logger.error(f"Vector store error: {vector_store_error}")
                # Return a response indicating the vector store is unavailable
                return {
                    "answer": "I'm currently unable to access the knowledge base. Please try again later.",
                    "sources": [],
                    "retrieved_count": 0,
                    "error": "vector_store_unavailable"
                }

            # Prepare context for the LLM from the search results
            context = []
            for result in search_results:
                context.append({
                    "content_snippet": result["content"],
                    "page_reference": result["metadata"].get("page_reference", ""),
                    "similarity_score": result["similarity_score"]
                })

            # Generate the answer using Cohere with error handling
            try:
                answer = self.cohere_service.generate_response(
                    prompt=question,
                    context=context
                )
            except Exception as llm_error:
                logger.error(f"LLM error: {llm_error}")
                return {
                    "answer": "I'm currently unable to generate a response. Please try again later.",
                    "sources": context,  # Still return context if available
                    "retrieved_count": len(search_results),
                    "error": "llm_unavailable"
                }

            # Return the answer with sources
            return {
                "answer": answer,
                "sources": context,
                "retrieved_count": len(search_results)
            }

        except Exception as e:
            logger.error(f"Unexpected error in robust RAG query: {e}")
            return {
                "answer": "An unexpected error occurred. Please try again later.",
                "sources": [],
                "retrieved_count": 0,
                "error": "unexpected_error"
            }

# Global instance
rag_service = RAGService();