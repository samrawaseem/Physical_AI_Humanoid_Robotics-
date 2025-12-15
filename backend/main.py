from fastapi import FastAPI
from src.config import settings
from src.api.v1.query import router as query_router
from src.api.v1.session import router as session_router
from src.utils.logger import app_logger
from src.middleware.request_logging import request_response_logging_middleware
from src.middleware.rate_limit import rate_limit_middleware

app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG-based question answering on book content",
    version="1.0.0"
)

# Add middleware (order matters)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(rate_limit_middleware)
app.middleware("http")(request_response_logging_middleware)

# Include API routes
app.include_router(query_router)
app.include_router(session_router)

@app.get("/")
async def root():
    app_logger.info("Root endpoint accessed")
    return {"message": "RAG Chatbot API is running"}

@app.get("/health")
async def health_check():
    app_logger.info("Health check endpoint accessed")
    return {"status": "healthy", "environment": settings.app_env}

if __name__ == "__main__":
    import uvicorn
    app_logger.info("Starting RAG Chatbot API server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development"
    )