from fastapi import HTTPException, status
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RAGException(HTTPException):
    """
    Base exception for RAG-related errors
    """
    def __init__(self, detail: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(status_code=status_code, detail=detail)
        logger.error(f"RAGException: {detail}")

class QueryProcessingError(RAGException):
    """
    Exception raised when there's an error processing a query
    """
    def __init__(self, detail: str = "Error processing query"):
        super().__init__(detail=detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VectorStoreError(RAGException):
    """
    Exception raised when there's an error with the vector store
    """
    def __init__(self, detail: str = "Vector store error"):
        super().__init__(detail=detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LLMError(RAGException):
    """
    Exception raised when there's an error with the language model
    """
    def __init__(self, detail: str = "Language model error"):
        super().__init__(detail=detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ValidationError(RAGException):
    """
    Exception raised when there's a validation error
    """
    def __init__(self, detail: str = "Validation error"):
        super().__init__(detail=detail, status_code=status.HTTP_400_BAD_REQUEST)

def handle_error(error: Exception, error_type: str = "general"):
    """
    General error handling function
    """
    logger.error(f"Error in {error_type}: {str(error)}")
    if isinstance(error, RAGException):
        raise error
    else:
        raise RAGException(detail=f"{error_type} error: {str(error)}")

# FastAPI exception handlers
def add_exception_handlers(app):
    """
    Add exception handlers to FastAPI app
    """
    @app.exception_handler(RAGException)
    async def handle_rag_exception(request, exc):
        return {
            "status": "error",
            "message": str(exc.detail),
            "status_code": exc.status_code
        }