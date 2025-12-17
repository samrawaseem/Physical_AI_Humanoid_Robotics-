import os
import sys
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Request

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Ensure backend directory is in sys.path
sys.path.append(BACKEND_DIR)
sys.path.append(BASE_DIR)

# Import the main app
try:
    from backend.main import app
    from backend.src.utils.logger import app_logger
except ImportError as e:
    # Fallback/Debug if imports fail (useful for debugging in HF logs)
    print(f"Import Error: {e}")
    # Try importing without backend prefix given sys.path adjustment
    from main import app
    from src.utils.logger import app_logger

app_logger.info(f"Setting up app.py serving...")
app_logger.info(f"Static Directory: {STATIC_DIR}")
app_logger.info(f"Backend Directory: {BACKEND_DIR}")

# Check if static directory exists
if os.path.exists(STATIC_DIR):
    # Mount static files to root
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    # Catch-all route for SPA client-side routing
    @app.exception_handler(404)
    async def not_found_exception_handler(request: Request, exc):
        if not request.url.path.startswith("/api"):
            index_path = os.path.join(STATIC_DIR, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
        return {"detail": "Not Found"}, 404
else:
    app_logger.warning(f"Static directory {STATIC_DIR} not found. Frontend will not be served.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app_logger.info(f"Starting Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
