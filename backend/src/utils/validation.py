import re
from typing import Optional, Union
from html import escape
import bleach
from pydantic import BaseModel, validator
from fastapi import HTTPException, status


def sanitize_input(text: str) -> str:
    """
    Sanitize input text by removing potentially harmful content
    """
    if not text:
        return text

    # Remove potentially dangerous HTML tags and attributes
    clean_text = bleach.clean(
        text,
        tags=['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li'],
        attributes={},
        strip=True
    )

    # Escape any remaining HTML entities
    clean_text = escape(clean_text)

    return clean_text


def validate_question(question: str) -> str:
    """
    Validate and sanitize a question input
    """
    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )

    # Check length
    if len(question) > 2000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is too long (maximum 2000 characters)"
        )

    # Check for potentially harmful content
    if re.search(r'<script|javascript:|vbscript:|onload|onerror', question, re.IGNORECASE):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question contains invalid characters or code"
        )

    # Sanitize the input
    sanitized_question = sanitize_input(question)

    return sanitized_question


def validate_selected_text(selected_text: Optional[str]) -> Optional[str]:
    """
    Validate and sanitize selected text input
    """
    if not selected_text:
        return selected_text

    # Check length
    if len(selected_text) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected text is too long (maximum 5000 characters)"
        )

    # Sanitize the input
    sanitized_text = sanitize_input(selected_text)

    return sanitized_text


def validate_page_content(page_content: Optional[str]) -> Optional[str]:
    """
    Validate and sanitize page content input
    """
    if not page_content:
        return page_content

    # Check length
    if len(page_content) > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page content is too long (maximum 10000 characters)"
        )

    # Sanitize the input
    sanitized_content = sanitize_input(page_content)

    return sanitized_content


def validate_session_id(session_id: Optional[str]) -> Optional[str]:
    """
    Validate session ID format
    """
    if not session_id:
        return session_id

    # Basic UUID validation pattern
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )

    if not uuid_pattern.match(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format"
        )

    return session_id


def is_valid_uuid(uuid_string: str) -> bool:
    """
    Check if a string is a valid UUID
    """
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(uuid_string))


def validate_content_length(content: str, max_length: int, field_name: str) -> str:
    """
    Generic function to validate content length
    """
    if len(content) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is too long (maximum {max_length} characters)"
        )

    return content


def escape_special_characters(text: str) -> str:
    """
    Escape special characters that could be used for injection attacks
    """
    if not text:
        return text

    # Escape special characters
    escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")

    return escaped_text


def validate_user_input(data: dict) -> dict:
    """
    Validate and sanitize all user input in a dictionary
    """
    validated_data = {}

    for key, value in data.items():
        if key == 'question' and isinstance(value, str):
            validated_data[key] = validate_question(value)
        elif key == 'selected_text' and isinstance(value, str):
            validated_data[key] = validate_selected_text(value)
        elif key == 'page_content' and isinstance(value, str):
            validated_data[key] = validate_page_content(value)
        elif key == 'session_id' and isinstance(value, str):
            validated_data[key] = validate_session_id(value)
        elif isinstance(value, str):
            # For other string fields, apply general sanitization
            validated_data[key] = sanitize_input(value)
        else:
            # For non-string fields, pass through unchanged
            validated_data[key] = value

    return validated_data


class ValidationError(Exception):
    """
    Custom validation error exception
    """
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


def validate_email(email: str) -> str:
    """
    Basic email validation
    """
    if not email:
        return email

    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValidationError("Invalid email format", "email")

    return email.lower().strip()


def validate_url(url: str) -> str:
    """
    Basic URL validation
    """
    if not url:
        return url

    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url):
        raise ValidationError("Invalid URL format", "url")

    return url


def clean_text_for_embedding(text: str) -> str:
    """
    Clean text specifically for embedding generation
    Removes or replaces characters that might interfere with embedding models
    """
    if not text:
        return text

    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text)

    # Remove control characters
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\t\n\r')

    # Limit length to reasonable size for embedding models
    if len(cleaned) > 8000:  # Most embedding models have context limits
        cleaned = cleaned[:8000]

    return cleaned.strip()