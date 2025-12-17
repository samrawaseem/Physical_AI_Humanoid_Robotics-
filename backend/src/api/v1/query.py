from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import uuid
import logging

from ...database import get_db
from ...models.chat_session import ChatSession
from ...models.chat_message import ChatMessage
from ...services.rag_service import rag_service
from ...services.session_service import session_service
from ...api.errors import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])

# Request model
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The user's question about the book content")
    selected_text: Optional[str] = Field(None, max_length=5000, description="Text selected by the user for context-specific queries")
    page_content: Optional[str] = Field(None, max_length=10000, description="Current page content for context-aware responses")

# Response models
class Source(BaseModel):
    content_snippet: str
    page_reference: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str
    message_id: str

class ErrorResponse(BaseModel):
    error: str
    code: str

@router.post("/query",
             response_model=QueryResponse,
             responses={
                 200: {"description": "Query processed successfully"},
                 400: {"description": "Invalid request parameters", "model": ErrorResponse},
                 500: {"description": "Internal server error", "model": ErrorResponse}
             })
async def query_endpoint(
    request: QueryRequest,
    session_id: Optional[str] = None,  # Allow passing an existing session ID
    db: Session = Depends(get_db)
):
    """
    Process user queries and return RAG-generated responses
    Accepts question and optional selected text context, queries the vector store,
    and generates responses using the language model
    """
    logger.info("=" * 80)
    logger.info("📨 INCOMING REQUEST to /api/v1/query")
    logger.info(f"Question: {request.question[:100]}..." if len(request.question) > 100 else f"Question: {request.question}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Has selected text: {bool(request.selected_text)}")
    logger.info(f"Has page content: {bool(request.page_content)}")
    logger.info("=" * 80)
    
    try:
        # Validate the request
        if not request.question.strip():
            logger.warning("⚠️ Empty question received")
            raise ValidationError("Question cannot be empty")

        logger.info("✅ Request validation passed")

        # Get or create session
        if session_id:
            logger.info(f"🔍 Looking for existing session: {session_id}")
            # Try to get existing session
            session = session_service.get_session(db, session_id)
            if not session:
                logger.info(f"❌ Session {session_id} not found, creating new session")
                # If session doesn't exist, create a new one
                session = session_service.create_session(db)
            else:
                logger.info(f"✅ Found existing session: {session_id}")
        else:
            logger.info("🆕 Creating new session")
            # Create a new session
            session = session_service.create_session(db)
        
        logger.info(f"📝 Session ID: {session.id}")

        # Create user message
        logger.info("💾 Saving user message to database")
        user_message = ChatMessage(
            session_id=session.id,
            sender="user",
            content=request.question,
            context={"selected_text": request.selected_text} if request.selected_text else {}
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        logger.info(f"✅ User message saved with ID: {user_message.id}")

        # Use RAG service to get the answer with robust error handling
        logger.info("🤖 Calling RAG service for answer generation")
        result = rag_service.robust_query(
            question=request.question,
            selected_text=request.selected_text,
            page_content=request.page_content
        )
        logger.info(f"✅ RAG service returned result")
        logger.info(f"Answer length: {len(result.get('answer', ''))} characters")
        logger.info(f"Number of sources: {len(result.get('sources', []))}")
        logger.info(f"Has error: {bool(result.get('error'))}")

        # Check if the result contains an error
        if result.get("error"):
            logger.warning(f"⚠️ RAG service returned error: {result['error']}")
            # Create bot response message even for error responses
            bot_message = ChatMessage(
                session_id=session.id,
                sender="bot",
                content=result["answer"],
                context={"sources": result.get("sources", [])}
            )
            db.add(bot_message)
            db.commit()
            db.refresh(bot_message)
            logger.info(f"💾 Bot error message saved with ID: {bot_message.id}")

            if result["error"] == "vector_store_unavailable":
                logger.error("❌ Vector store unavailable")
                # Return a specific error response for vector store issues
                return QueryResponse(
                    answer=result["answer"],
                    sources=[],
                    session_id=str(session.id),
                    message_id=str(bot_message.id)
                )
            elif result["error"] == "llm_unavailable":
                logger.error("❌ LLM unavailable")
                # Return a specific error response for LLM issues
                return QueryResponse(
                    answer=result["answer"],
                    sources=result.get("sources", []),
                    session_id=str(session.id),
                    message_id=str(bot_message.id)
                )
            else:
                logger.error(f"❌ Other error: {result['error']}")
                # For other errors, return the result as is
                return QueryResponse(
                    answer=result["answer"],
                    sources=result.get("sources", []),
                    session_id=str(session.id),
                    message_id=str(bot_message.id)
                )

        # Create bot response message for successful responses
        logger.info("💾 Saving bot response to database")
        bot_message = ChatMessage(
            session_id=session.id,
            sender="bot",
            content=result["answer"],
            context={"sources": result["sources"]}
        )
        db.add(bot_message)
        db.commit()
        logger.info(f"✅ Bot message saved with ID: {bot_message.id}")

        # Format the response
        logger.info("📦 Formatting response")
        response = QueryResponse(
            answer=result["answer"],
            sources=[
                Source(
                    content_snippet=source["content_snippet"],
                    page_reference=source["page_reference"],
                    similarity_score=source["similarity_score"]
                )
                for source in result["sources"]
            ],
            session_id=str(session.id),
            message_id=str(bot_message.id)
        )

        logger.info("✅ SUCCESS - Returning response")
        logger.info("=" * 80)
        return response

    except ValidationError as e:
        logger.warning(f"⚠️ Validation error in query endpoint: {e.detail}")
        logger.info("=" * 80)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e.detail))

    except Exception as e:
        logger.error(f"❌ EXCEPTION in query endpoint: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while processing the query"
        )