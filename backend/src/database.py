from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from .config import settings
import logging

logger = logging.getLogger(__name__)

# Create the database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
    echo=(settings.log_level.lower() == 'debug')  # Log SQL queries in debug mode
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for declarative models
Base = declarative_base()

def get_db():
    """
    Dependency function to get database session for FastAPI
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    """
    Context manager for database sessions
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """
    Initialize the database by creating all tables
    """
    logger.info("Initializing database tables...")
    # Import models to ensure they are registered with Base
    from .models.chat_session import ChatSession
    from .models.chat_message import ChatMessage
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")

if __name__ == "__main__":
    init_db()