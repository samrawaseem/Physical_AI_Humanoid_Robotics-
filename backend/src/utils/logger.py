import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = "rag_chatbot",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a comprehensive logger for the RAG Chatbot application
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        # Ensure the log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "rag_chatbot") -> logging.Logger:
    """
    Get a logger instance with the specified name
    """
    return logging.getLogger(name)


# Create the main application logger
app_logger = setup_logger(
    name="rag_chatbot",
    log_file="logs/app.log",
    log_level=logging.INFO
)


def log_api_request(
    endpoint: str,
    method: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    query_length: Optional[int] = None
):
    """
    Log API requests with relevant information
    """
    app_logger.info(
        f"API Request: {method} {endpoint} | "
        f"User: {user_id or 'unknown'} | "
        f"Session: {session_id or 'none'} | "
        f"Query length: {query_length or 0} chars"
    )


def log_api_response(
    endpoint: str,
    status_code: int,
    response_time_ms: float,
    response_length: Optional[int] = None
):
    """
    Log API responses with relevant information
    """
    app_logger.info(
        f"API Response: {endpoint} | "
        f"Status: {status_code} | "
        f"Time: {response_time_ms:.2f}ms | "
        f"Response length: {response_length or 0} chars"
    )


def log_error(
    error: Exception,
    context: str = "",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Log errors with context information
    """
    app_logger.error(
        f"Error in {context} | "
        f"User: {user_id or 'unknown'} | "
        f"Session: {session_id or 'none'} | "
        f"Error: {str(error)} | "
        f"Type: {type(error).__name__}",
        exc_info=True  # Include traceback
    )


def log_vector_store_operation(
    operation: str,
    collection_name: str,
    items_count: Optional[int] = None,
    success: bool = True
):
    """
    Log vector store operations
    """
    status = "SUCCESS" if success else "FAILED"
    count_info = f" | Items: {items_count}" if items_count is not None else ""
    app_logger.info(
        f"Vector Store: {operation} | Collection: {collection_name} | Status: {status}{count_info}"
    )


def log_embedding_generation(
    text_length: int,
    model_used: str,
    success: bool = True
):
    """
    Log embedding generation operations
    """
    status = "SUCCESS" if success else "FAILED"
    app_logger.info(
        f"Embedding Generation | Text length: {text_length} | Model: {model_used} | Status: {status}"
    )


def log_llm_call(
    prompt_length: int,
    model_used: str,
    response_length: int,
    success: bool = True
):
    """
    Log LLM API calls
    """
    status = "SUCCESS" if success else "FAILED"
    app_logger.info(
        f"LLM Call | Model: {model_used} | Prompt length: {prompt_length} | "
        f"Response length: {response_length} | Status: {status}"
    )