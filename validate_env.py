#!/usr/bin/env python3
"""
Environment Variables Validation Script
Checks that all required environment variables are properly set
"""

import os
import sys
from urllib.parse import urlparse

def validate_env_vars():
    """Validate all environment variables in the .env file"""
    print("Validating environment variables...")

    # Read the .env file
    env_file = "backend/.env"
    if not os.path.exists(env_file):
        print(f"[ERROR] {env_file} not found")
        return False

    with open(env_file, 'r') as f:
        env_content = f.read()

    # Check for placeholder values
    issues = []

    if "your_qdrant_api_key_here" in env_content:
        issues.append("QDRANT_API_KEY still has placeholder value")

    if "your_cohere_api_key_here" in env_content:
        issues.append("COHERE_API_KEY still has placeholder value")

    if "your_neon_db_password_here" in env_content:
        issues.append("NEON_DB_PASSWORD still has placeholder value")

    # Check for actual values that should be set
    env_vars = {}
    for line in env_content.split('\n'):
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            env_vars[key.strip()] = value.strip().strip("'\"")

    # Validate URLs
    if 'QDRANT_URL' in env_vars:
        qdrant_url = env_vars['QDRANT_URL']
        if qdrant_url and not qdrant_url.startswith(('http://', 'https://')):
            issues.append(f"QDRANT_URL should start with http:// or https://: {qdrant_url}")

    if 'DATABASE_URL' in env_vars:
        db_url = env_vars['DATABASE_URL']
        try:
            parsed = urlparse(db_url)
            if parsed.scheme != 'postgresql':
                issues.append(f"DATABASE_URL should use postgresql scheme: {db_url}")
        except Exception as e:
            issues.append(f"Invalid DATABASE_URL format: {db_url} - {e}")

    # Report issues
    if issues:
        print("[ERROR] Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("[SUCCESS] All environment variables are properly configured!")
        return True

if __name__ == "__main__":
    success = validate_env_vars()
    if not success:
        sys.exit(1)