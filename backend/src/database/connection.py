import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

# Database URL - defaults to SQLite in data directory
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/eioku.db")


def _create_engine(database_url: str):
    """Create database engine with appropriate configuration for the database type."""
    if database_url.startswith("postgresql"):
        # PostgreSQL configuration with connection pooling
        # Pool size accounts for: 16 worker threads + API requests + overhead
        return create_engine(
            database_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=300,  # Recycle connections after 5 minutes
        )
    elif database_url.startswith("sqlite"):
        # SQLite configuration - disable thread check for multi-threaded access
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )
    else:
        # Generic configuration for other databases
        return create_engine(database_url)


# Create engine
engine = _create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Thread-safe scoped session for worker pools
ScopedSession = scoped_session(SessionLocal)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_scoped_db():
    """Get thread-local database session for worker pools."""
    return ScopedSession()


def remove_scoped_session():
    """Remove the current scoped session (call at end of request/task)."""
    ScopedSession.remove()
