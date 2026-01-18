#!/bin/bash

# Create pull request for Task 3 - Database Access Layer implementation

gh pr create \
  --title "feat(database): implement Task 3 - Database Access Layer with clean architecture" \
  --body "## Summary

This PR implements Task 3 from the implementation plan - Database Access Layer for the Eioku semantic video search platform. Building on the clean architecture foundation from Task 2, this adds comprehensive data access patterns and business logic layers.

## Changes Made

### Clean Architecture Implementation
- **Domain Models** - Pure business objects decoupled from persistence
- **Repository Pattern** - Abstract interfaces with SQLAlchemy implementations
- **Service Layer** - Business logic and domain operations
- **API Controllers** - FastAPI REST endpoints with OpenAPI documentation
- **Dependency Injection** - Proper IoC for testability and modularity

### Video Management (Task 3.1)
- **VideoRepository** - Complete CRUD operations with domain/entity mapping
- **VideoService** - Business logic for video lifecycle management
- **Video API** - Full REST endpoints for video operations
- **File Hash Support** - Ready for discovery process integration

### Database Features
- **Modern SQLAlchemy 2.0** - Latest patterns and best practices
- **Alembic Migrations** - Database schema versioning and evolution
- **Connection Management** - Proper session handling and dependency injection
- **Performance Indexes** - Optimized queries on frequently accessed columns

### API Features
- **OpenAPI 3.0 Spec** - Auto-generated documentation at /docs and /redoc
- **JSON Schema Validation** - Automatic request/response validation with detailed errors
- **REST Endpoints** - Clean, consistent API design following HTTP standards
- **Error Handling** - Proper HTTP status codes and structured error responses
- **No Hardcoded Prefixes** - Reverse proxy handles /api routing

## API Endpoints

### Video Management
- \`POST /v1/videos/\` - Create video for processing
- \`GET /v1/videos/{id}\` - Get video by ID
- \`GET /v1/videos/\` - List videos (filterable by status)
- \`PATCH /v1/videos/{id}\` - Update video metadata/status
- \`DELETE /v1/videos/{id}\` - Delete video and associated data

## Architecture Benefits

### For Orchestrator (Internal)
- **Direct Service Access** - No HTTP overhead for internal operations
- **Type Safety** - Full TypeScript-like type checking with modern Python
- **Business Logic** - Rich domain models with validation and business rules
- **Transaction Support** - Proper database transaction management

### For External Clients
- **REST API** - Standard HTTP interface with comprehensive documentation
- **OpenAPI Integration** - Auto-generated client SDKs and documentation
- **JSON Validation** - Automatic request/response validation
- **Error Handling** - Structured error responses with detailed messages

## Testing

- ✅ **24 comprehensive tests** - Full coverage including API endpoints
- ✅ **Database isolation** - Each test uses temporary database
- ✅ **Integration tests** - End-to-end API testing with real database
- ✅ **Unit tests** - Domain models, services, and repositories
- ✅ **Quality gates** - Ruff formatting and linting pass

## Requirements Traceability

This implementation covers:
- **Task 3.1** - Video DAO with CRUD operations and query methods
- **Task 3.9** - Database connection management with session handling
- **Task 3.10** - Comprehensive unit tests for data access layer
- **Requirements 4.1, 4.2, 4.3** - Video management and query operations

## Technical Details

### Domain-Driven Design
- **Pure domain models** - No persistence concerns in business logic
- **Repository abstraction** - Interface-based data access for testability
- **Service layer** - Encapsulates business rules and workflows
- **Clean boundaries** - Clear separation between layers

### Modern Python Patterns
- **Type annotations** - Full type safety with X | Y union syntax
- **Dependency injection** - FastAPI's built-in DI container
- **Async support** - Ready for async operations when needed
- **Error handling** - Structured exceptions with proper HTTP mapping

## Next Steps

Ready for:
1. **Task 3.2-3.8** - Additional repository implementations (Transcription, Scene, Object, Face, Topic, PathConfig, Task)
2. **Task 4** - Path management and video discovery
3. **Task 5** - Task orchestration system
4. **Integration** - Connect with video processing pipeline

## Deployment Notes

- **Database migrations** - Run automatically on application startup
- **Environment configuration** - Database URL configurable via environment variables
- **Reverse proxy ready** - No hardcoded API prefixes
- **Container friendly** - Works with Docker and container orchestration" \
  --head feature/task-3-database-access-layer \
  --base main
