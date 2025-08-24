import logging
import uvicorn
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load environment variables first
from dotenv import load_dotenv
project_root = Path(__file__).parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Now import settings after loading .env
from src.core.config import settings
from src.services.cache_service import CacheService

# Configure logging
level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
logging.basicConfig(
    level=level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global cache service for cleanup
cache_service = CacheService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Starting RAG Agent application")
    yield
    # Shutdown
    logger.info("Shutting down RAG Agent application")
    await cache_service.close()


# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent",
    description="Enterprise RAG (Retrieval-Augmented Generation) Agent with LangGraph workflow orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files (create directory if it doesn't exist)
import os
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("templates"):
    os.makedirs("templates")
    
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Include API routes (import here to avoid early initialization)
from src.api.routes import router as api_router
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Web UI home page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "RAG Agent"}
    )


@app.get("/docs-redirect")
async def docs_redirect():
    """Redirect to API documentation"""
    return {"message": "API documentation available at /docs"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
