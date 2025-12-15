from fastapi import Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
import logging
from src.utils.logger import log_api_request, log_api_response

logger = logging.getLogger(__name__)

async def request_response_logging_middleware(request: Request, call_next):
    """
    Middleware to log API requests and responses
    """
    start_time = time.time()

    # Get user ID and session ID from headers or other sources
    user_id = request.headers.get("user-id", "anonymous")
    session_id = request.headers.get("session-id")

    # Log the incoming request
    query_length = 0
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body_bytes = await request.body()
            query_length = len(body_bytes)
        except Exception:
            # If we can't read the body, continue without it
            pass

    log_api_request(
        endpoint=request.url.path,
        method=request.method,
        user_id=user_id,
        session_id=session_id,
        query_length=query_length
    )

    # Process the request
    response = await call_next(request)

    # Calculate response time
    response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Log the response
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    # Set the body iterator back for the response to be sent
    async def iterator():
        yield response_body
    response.body_iterator = iterator()

    log_api_response(
        endpoint=request.url.path,
        status_code=response.status_code,
        response_time_ms=response_time,
        response_length=len(response_body)
    )

    return response