#!/bin/bash

# Create pull request for database schema implementation

gh pr create \
  --title "feat(database): Complete database schema implementation with 8 tables and migrations" \
  --body "## Summary

This PR implements the complete database schema for the Eioku semantic video search platform, covering Task 2 (Create database schema and migrations) from the implementation plan.

## Changes Made

### Database Tables Implemented
- **Videos table** - Core video metadata with file paths, duration, processing status
- **Transcriptions table** - Speech-to-text segments with timestamps and speaker identification  
- **Scenes table** - Scene boundaries with thumbnails for video navigation
- **Objects table** - Object detection results with bounding boxes and timestamps
- **Faces table** - Face detection with person clustering and confidence scores
- **Topics table** - Topic extraction with keywords and relevance scores
- **PathConfigs table** - Video source path configuration with recursive scanning
- **Tasks table** - Processing task queue with dependencies and status tracking

### Infrastructure
- **SQLAlchemy 2.0** models with proper relationships and constraints
- **Alembic migrations** for each table with upgrade/downgrade support
- **Migration runner** that executes on FastAPI startup
- **Modern FastAPI patterns** using lifespan events instead of deprecated on_event
- **Comprehensive testing** with 12 tests covering all models and migrations

### Key Features
- **Foreign key relationships** between all tables and Videos
- **Performance indexes** on frequently queried columns (video_id, status, labels)
- **JSON fields** for complex data (timestamps, bounding boxes, keywords, dependencies)
- **Unique constraints** to prevent duplicate entries
- **Audit timestamps** for created_at and updated_at tracking
- **Environment configuration** for database URL override

## Testing

- ✅ **12 tests passing** - All models and migrations tested
- ✅ **Code quality** - Ruff linting passes, follows PEP 8
- ✅ **Migration validation** - All migrations can upgrade and downgrade
- ✅ **Foreign key integrity** - Relationships properly enforced
- ✅ **JSON field support** - Complex data structures work correctly

## Requirements Traceability

This implementation covers:
- **Task 2.1-2.8** - All 8 database tables with proper schema
- **Task 2.9** - Migration scripts and schema versioning
- **Requirements 1.1, 1.4, 1.5, 4.1, 4.4, 6.2, 6.5, 7.1, 7.2, 9.1, 9.2, 10.4, 10.6, 10.7, 10.9, 10.10**

## Next Steps

Ready for Task 3: Implement database access layer (DAOs) for CRUD operations on each table.

## Deployment Notes

- Database migrations run automatically on application startup
- SQLite default with environment variable override support
- No breaking changes to existing API endpoints" \
  --head feature/task-1-implementation \
  --base main
