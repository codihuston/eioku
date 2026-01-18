from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.video_controller import router as video_router
from src.database.migrations import run_migrations


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    # Startup
    run_migrations()
    yield
    # Shutdown (nothing to do for now)


app = FastAPI(
    title="Eioku - Semantic Video Search API",
    description="API for semantic video search and processing",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(video_router, prefix="/v1")


@app.get("/")
async def root():
    """Hello world endpoint."""
    return {"message": "Eioku API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
