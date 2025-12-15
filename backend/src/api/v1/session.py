from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
import logging

from ...database import get_db
from ...models.chat_session import ChatSession
from ...models.chat_message import ChatMessage
from ...services.session_service import session_service
from ...api.errors import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["session"])

# Response models
class MessageResponse(BaseModel):
    id: str
    session_id: str
    sender: str
    content: str
    timestamp: str
    context: dict

class SessionResponse(BaseModel):
    id: str
    created_at: str
    updated_at: str
    user_id: str
    messages: List[MessageResponse]

@router.get("/session/{session_id}",
            response_model=SessionResponse,
            responses={
                200: {"description": "Session retrieved successfully"},
                404: {"description": "Session not found"}
            })
async def get_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve a chat session with its messages
    """
    try:
        # Get the session
        session = session_service.get_session_with_messages(db, session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        # Get messages for the session
        messages = session_service.get_session_messages(db, session_id)

        # Format the response
        session_response = SessionResponse(
            id=str(session.id),
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            user_id=session.user_id or "",
            messages=[
                MessageResponse(
                    id=str(msg.id),
                    session_id=str(msg.session_id),
                    sender=msg.sender,
                    content=msg.content,
                    timestamp=msg.timestamp.isoformat(),
                    context=msg.context
                )
                for msg in messages
            ]
        )

        return session_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while retrieving the session"
        )