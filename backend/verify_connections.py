import sys
import os
import logging

# Ensure we can import from src
sys.path.append(os.getcwd())

import uvicorn
from fastapi import FastAPI
from src.config import settings
from qdrant_client import QdrantClient
import cohere
from sqlalchemy import create_engine, text

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verification")

def check_env_vars():
    logger.info("--- Checking Environment Variables ---")
    vars_to_check = [
        "QDRANT_URL", "QDRANT_API_KEY", 
        "COHERE_API_KEY", 
        "DATABASE_URL"
    ]
    
    for var in vars_to_check:
        value = settings.get_env_var(var) if hasattr(settings, 'get_env_var') else os.getenv(var)
        # Using settings object directly as fallback or primary if available
        if var == "QDRANT_URL": value = settings.qdrant_url
        if var == "QDRANT_API_KEY": value = settings.qdrant_api_key
        if var == "COHERE_API_KEY": value = settings.cohere_api_key
        if var == "DATABASE_URL": value = settings.database_url

        if value:
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            logger.info(f"✅ {var}: Found ({masked})")
        else:
            logger.error(f"❌ {var}: Missing or Empty")

def check_qdrant():
    logger.info("\n--- Checking Qdrant Connection ---")
    try:
        if settings.qdrant_api_key:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                port=settings.qdrant_port if settings.qdrant_port != 443 else None,
                https=True if settings.qdrant_port == 443 else False
            )
        else:
            client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        
        collections = client.get_collections()
        logger.info(f"✅ Qdrant Connected! Found {len(collections.collections)} collections.")
        for col in collections.collections:
            logger.info(f"   - Collection: {col.name}")
            
    except Exception as e:
        logger.error(f"❌ Qdrant Connection Failed: {e}")

def check_cohere():
    logger.info("\n--- Checking Cohere Connection ---")
    try:
        client = cohere.Client(settings.cohere_api_key)
        # Try a simple embedding generation
        response = client.embed(
            texts=["Hello world"],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        if response.embeddings:
            logger.info("✅ Cohere Connected and generated embedding!")
        else:
            logger.warning("⚠️ Cohere connected but returned no embeddings.")
    except Exception as e:
        logger.error(f"❌ Cohere Connection Failed: {e}")

def check_neon_db():
    logger.info("\n--- Checking Neon DB Connection ---")
    try:
        engine = create_engine(settings.database_url)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Database (Neon) Connected! Query 'SELECT 1' successful.")
    except Exception as e:
        logger.error(f"❌ Database (Neon) Connection Failed: {e}")

if __name__ == "__main__":
    logger.info("Starting Verification Process...")
    check_env_vars()
    check_qdrant()
    check_cohere()
    check_neon_db()
    logger.info("\nVerification Complete.")
