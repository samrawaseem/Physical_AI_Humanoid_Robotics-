from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

def get_env_var(var_name: str, default_value: str = "") -> str:
    """Get environment variable, returning default if it's a placeholder value"""
    value = os.getenv(var_name, default_value)
    # Check if the value is a placeholder (contains common placeholder patterns)
    if value and ('your_' in value or 'placeholder' in value.lower() or
                  value == 'port' or 'password' in value or
                  'username' in value or 'host' in value or
                  'dbname' in value or 'api_key' in value or
                  'url' in value):
        return default_value
    return value

def get_env_int_var(var_name: str, default_value: int) -> int:
    """Get environment variable as integer, returning default if invalid"""
    value_str = get_env_var(var_name, str(default_value))
    try:
        return int(value_str)
    except ValueError:
        return default_value

class Settings(BaseSettings):
    # Qdrant Configuration
    qdrant_url: Optional[str] = get_env_var("QDRANT_URL")
    qdrant_api_key: Optional[str] = get_env_var("QDRANT_API_KEY")
    qdrant_host: str = get_env_var("QDRANT_HOST", "localhost")
    qdrant_port: int = get_env_int_var("QDRANT_PORT", 6333)

    # Cohere Configuration
    cohere_api_key: str = get_env_var("COHERE_API_KEY", "")
    cohere_model: str = get_env_var("COHERE_MODEL", "command-r-plus")

    # Database Configuration
    database_url: str = get_env_var("DATABASE_URL", "postgresql://localhost:5432/mybook")  # Default to local DB
    neon_db_name: Optional[str] = get_env_var("NEON_DB_NAME")
    neon_db_user: Optional[str] = get_env_var("NEON_DB_USER")
    neon_db_password: Optional[str] = get_env_var("NEON_DB_PASSWORD")
    neon_db_host: Optional[str] = get_env_var("NEON_DB_HOST")

    # Application Configuration
    app_env: str = get_env_var("APP_ENV", "development")
    api_key: Optional[str] = get_env_var("API_KEY")
    log_level: str = get_env_var("LOG_LEVEL", "info")

    # RAG Configuration
    vector_collection_name: str = get_env_var("VECTOR_COLLECTION_NAME", "book_content")
    embedding_dimension: int = get_env_int_var("EMBEDDING_DIMENSION", 1024)
    max_context_tokens: int = get_env_int_var("MAX_CONTEXT_TOKENS", 3000)

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env file

# Create a singleton instance of settings
settings = Settings()