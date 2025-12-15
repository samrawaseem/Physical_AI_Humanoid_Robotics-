import os
import sys
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Request

# Add the current directory to python path to ensure imports work
# We need to add the 'backend' directory to sys.path so that 'src' imports in main.py work
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)
# Also add root for good measure if needed, but backend_dir is critical for 'src'
sys.path.append(os.path.join(backend_dir, '..'))

# Import the main app from backend.main
# We assume backend/main.py initializes 'app'
from backend.main import app
from src.utils.logger import app_logger

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The static folder is at the root level in the Docker container (./static)
# But since we are running from backend/, and BASE_DIR is .../backend
# We need to go up one level to find 'static' which we copied to the root in Dockerfile
STATIC_DIR = os.path.join(BASE_DIR, '..', 'static')

app_logger.info(f"Setting up static file serving from: {STATIC_DIR}")

# Check if static directory exists
if os.path.exists(STATIC_DIR):
    # Mount static files
    # We mount it to root "/" to serve index.html and assets
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    # Catch-all route for SPA client-side routing (redirect 404s to index.html)
    # Note: Because we mounted StaticFiles at root, it handles valid files.
    # We need an exception handler or a catch-all for non-existant paths to return index.html
    
    # However, FastAPI's StaticFiles takes precedence. 
    # To support clean URLs for Docusaurus, we usually rely on the build structure.
    # Docusaurus builds 'path/index.html' for '/path'.
    # So StaticFiles with html=True usually handles it.
    
    # But for truly dynamic routes or deep links that might not map to a file:
    @app.exception_handler(404)
    async def not_found_exception_handler(request: Request, exc):
        # Only return index.html for non-API routes
        if not request.url.path.startswith("/api"):
            index_path = os.path.join(STATIC_DIR, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
        return {"detail": "Not Found"}, 404
        
else:
    app_logger.warning(f"Static directory {STATIC_DIR} not found. Frontend will not be served.")

if __name__ == "__main__":
    # Hugging Face Spaces sets PORT env var
    port = int(os.environ.get("PORT", 7860))
    app_logger.info(f"Starting Deployment Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
