import os
import sys
from typing import List

def validate_environment() -> List[str]:
    """
    Validates that required environment variables are set.
    Returns a list of missing environment variables.
    """
    required_vars = [
        "COHERE_API_KEY",
        "DATABASE_URL"
    ]

    # Qdrant URL is required unless using local instance
    if not os.getenv("QDRANT_URL") and not (os.getenv("QDRANT_HOST") and os.getenv("QDRANT_PORT")):
        required_vars.append("QDRANT_URL or (QDRANT_HOST and QDRANT_PORT)")

    missing_vars = []
    for var in required_vars:
        # Handle special case for QDRant
        if var == "QDRANT_URL or (QDRANT_HOST and QDRANT_PORT)":
            if not os.getenv("QDRANT_URL") and not (os.getenv("QDRANT_HOST") and os.getenv("QDRANT_PORT")):
                missing_vars.append("QDRANT_URL or both QDRANT_HOST and QDRANT_PORT")
        elif not os.getenv(var):
            missing_vars.append(var)

    return missing_vars

def check_environment() -> bool:
    """
    Checks environment variables and exits if any required ones are missing.
    """
    missing_vars = validate_environment()

    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("Refer to .env.example for required variables.")
        sys.exit(1)

    print("All required environment variables are set.")
    return True

if __name__ == "__main__":
    check_environment()