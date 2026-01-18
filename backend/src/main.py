from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.database.migrations import run_migrations


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    # Startup
    run_migrations()
    yield
    # Shutdown (nothing to do for now)


app = FastAPI(title="Eioku", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def root():
    """Hello world endpoint."""
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
