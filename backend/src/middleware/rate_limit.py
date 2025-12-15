from fastapi import Request, HTTPException, status
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests: int = 10, window: int = 60):
        """
        Initialize rate limiter
        :param requests: Number of requests allowed
        :param window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.requests_log = defaultdict(lambda: deque())

    def check_limit(self, identifier: str) -> bool:
        """
        Check if the identifier has exceeded the rate limit
        :param identifier: Unique identifier for the client (e.g., IP address)
        :return: True if within limit, False if exceeded
        """
        now = time.time()
        request_log = self.requests_log[identifier]

        # Remove old requests outside the time window
        while request_log and request_log[0] <= now - self.window:
            request_log.popleft()

        # Check if the limit is exceeded
        if len(request_log) >= self.requests:
            return False

        # Add current request
        request_log.append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter(requests=30, window=60)  # 30 requests per minute per IP


async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware to limit the rate of requests
    """
    # Get client IP address
    client_ip = request.client.host

    # Check rate limit
    if not rate_limiter.check_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

    # Continue with the request
    response = await call_next(request)
    return response