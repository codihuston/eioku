# Implementation Plan: Artifact Envelope Architecture

## Overview

This plan implements the artifact envelope architecture for Eioku's video processing system. The implementation follows an incremental approach, building the core infrastructure first, then migrating each artifact type, and finally adding API endpoints and projections.

## Tasks

- [x] 1. Database Schema and Migrations
- [x] 1.1 Create Alembic migration for artifacts table (PostgreSQL with JSONB)
  - Create artifacts table with all envelope fields
  - Add composite indexes for common query patterns
  - Add GIN index for JSONB payload
  - _Requirements: 1.1, 1.4, 1.5_

- [x] 1.2 Create Alembic migration for runs table
  - Create runs table for pipeline execution tracking
  - Add indexes for asset_id and status
  - _Requirements: 2.1_

- [x] 1.3 Create Alembic migration for artifact_selections table
  - Create selection policies table
  - Add primary key on (asset_id, artifact_type)
  - _Requirements: 6.1_

- [ ] 2. Core Artifact Infrastructure
- [ ] 2.1 Implement ArtifactEnvelope domain model
  - Create dataclass with all envelope fields
  - Add validation for required fields
  - _Requirements: 1.1, 12.1, 12.2, 12.3_

- [ ] 2.2 Implement SchemaRegistry
  - Create registry class with register/get_schema methods
  - Add validation method using Pydantic
  - Raise SchemaNotFoundError for unregistered schemas
  - _Requirements: 3.1, 3.4_

- [ ]* 2.3 Write property test for schema registry lookup
  - **Property 5: Schema Registry Lookup**
  - **Validates: Requirements 3.1, 3.4**

- [ ] 2.4 Implement artifact type schemas (Pydantic models)
  - TranscriptSegmentV1
  - SceneV1
  - ObjectDetectionV1
  - FaceDetectionV1
  - PlaceClassificationV1
  - OcrTextV1
  - _Requirements: 4.1, 13.1, 14.1, 15.1, 16.1, 17.1_

- [ ] 2.5 Register all schemas at application startup
  - Add schema registration to app initialization
  - _Requirements: 3.5_

- [ ]* 2.6 Write property test for schema validation round-trip
  - **Property 2: Schema Validation Round-Trip**
  - **Validates: Requirements 1.2, 3.2, 3.3**

- [ ] 3. Artifact Repository
- [ ] 3.1 Implement ArtifactRepository with CRUD operations
  - create() with schema validation
  - get_by_id()
  - get_by_asset() with filtering
  - get_by_span() for time range queries
  - delete()
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 3.2 Implement selection policy application in repository
  - _apply_selection_policy() helper method
  - Support all selection modes (default, pinned, latest, profile, best_quality)
  - _Requirements: 7.4, 6.3, 6.4, 6.5_

- [ ]* 3.3 Write property test for artifact envelope completeness
  - **Property 1: Artifact Envelope Completeness**
  - **Validates: Requirements 1.1, 12.1, 12.2, 12.3**

- [ ]* 3.4 Write property test for selection policy application
  - **Property 9: Selection Policy Application**
  - **Validates: Requirements 6.3, 6.4, 6.5, 7.4, 11.3**

- [ ]* 3.5 Write property test for time span overlap query
  - **Property 15: Time Span Overlap Query**
  - **Validates: Requirements 7.3**

- [ ] 4. Run Tracking
- [ ] 4.1 Implement Run domain model and repository
  - Create Run dataclass
  - Implement RunRepository with CRUD operations
  - _Requirements: 2.1_

- [ ]* 4.2 Write property test for run lifecycle tracking
  - **Property 4: Run Lifecycle Tracking**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

- [ ] 5. Selection Policy Manager
- [ ] 5.1 Implement SelectionPolicy domain model
  - Create dataclass with all policy fields
  - _Requirements: 6.1_

- [ ] 5.2 Implement SelectionPolicyManager
  - get_policy() method
  - set_policy() method
  - get_default_policy() method
  - _Requirements: 6.2_

- [ ] 6. Checkpoint - Core Infrastructure Complete
- Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Transcript Artifact Integration
- [ ] 7.1 Update transcription service to create transcript.segment artifacts
  - Modify WhisperService or equivalent to use ArtifactRepository
  - Create artifacts with proper envelope metadata
  - _Requirements: 4.2, 4.3_

- [ ]* 7.2 Write property test for artifact type storage consistency (transcripts)
  - **Property 6: Artifact Type Storage Consistency (partial)**
  - **Validates: Requirements 4.2**

- [ ] 7.3 Create Alembic migration for transcript_fts projection table
  - Create table with tsvector column
  - Add GIN index on text_tsv
  - Add index on (asset_id, start_ms)
  - _Requirements: 5.1_

- [ ] 7.4 Implement transcript FTS projection synchronization
  - Create sync_transcript_fts() function
  - Call from artifact creation
  - _Requirements: 5.2_

- [ ]* 7.5 Write property test for projection synchronization (transcripts)
  - **Property 7: Projection Synchronization (partial)**
  - **Validates: Requirements 5.2**

- [ ]* 7.6 Write property test for FTS search correctness (transcripts)
  - **Property 8: FTS Search Correctness (partial)**
  - **Validates: Requirements 5.3, 5.4**

- [ ] 8. Scene Artifact Integration
- [ ] 8.1 Update scene detection service to create scene artifacts
  - Modify PySceneDetect integration to use ArtifactRepository
  - _Requirements: 13.2, 13.3_

- [ ] 8.2 Create Alembic migration for scene_ranges projection table
  - _Requirements: 20.1_

- [ ] 8.3 Implement scene_ranges projection synchronization
  - _Requirements: 20.2_

- [ ]* 8.4 Write property test for artifact storage (scenes)
  - **Property 6: Artifact Type Storage Consistency (partial)**
  - **Validates: Requirements 13.2**

- [ ] 9. Object Detection Artifact Integration
- [ ] 9.1 Update object detection service to create object.detection artifacts
  - Modify YOLO integration to use ArtifactRepository
  - Create one artifact per detection
  - _Requirements: 14.2, 14.3, 14.5_

- [ ] 9.2 Create Alembic migration for object_labels projection table
  - _Requirements: 18.1_

- [ ] 9.3 Implement object_labels projection synchronization
  - _Requirements: 18.2_

- [ ]* 9.4 Write property test for artifact storage (objects)
  - **Property 6: Artifact Type Storage Consistency (partial)**
  - **Validates: Requirements 14.2**

- [x] 10. Face Detection Artifact Integration
- [x] 10.1 Update face detection service to create face.detection artifacts
  - Modify face detection integration to use ArtifactRepository
  - _Requirements: 15.2, 15.3_

- [x] 10.2 Create Alembic migration for face_clusters projection table
  - _Requirements: 19.1_

- [x] 10.3 Implement face_clusters projection synchronization
  - _Requirements: 19.2_

- [ ]* 10.4 Write property test for artifact storage (faces)
  - **Property 6: Artifact Type Storage Consistency (partial)**
  - **Validates: Requirements 15.2**

- [ ] 11. Place Classification Artifact Integration
- [ ] 11.1 Update place detection service to create place.classification artifacts
  - Modify ResNet Places365 integration to use ArtifactRepository
  - _Requirements: 16.2, 16.3_

- [ ]* 11.2 Write property test for artifact storage (places)
  - **Property 6: Artifact Type Storage Consistency (partial)**
  - **Validates: Requirements 16.2**

- [ ] 12. OCR Text Detection Artifact Integration
- [ ] 12.1 Update OCR service to create ocr.text artifacts
  - Modify EasyOCR integration to use ArtifactRepository
  - _Requirements: 17.2, 17.3_

- [ ] 12.2 Create Alembic migration for ocr_fts projection table
  - Create table with tsvector column
  - Add GIN index on text_tsv
  - _Requirements: 21.1_

- [ ] 12.3 Implement ocr_fts projection synchronization
  - _Requirements: 21.2_

- [ ]* 12.4 Write property test for artifact storage (OCR)
  - **Property 6: Artifact Type Storage Consistency (partial)**
  - **Validates: Requirements 17.2**

- [ ] 13. Checkpoint - All Artifact Types Migrated
- Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Jump Navigation Service
- [ ] 14.1 Implement JumpNavigationService
  - jump_next() method
  - jump_prev() method
  - _filter_artifacts() helper
  - Support all artifact types (scene, transcript, object, face, place, ocr)
  - _Requirements: 8.2, 8.3, 22.1, 22.2, 22.3, 22.4, 22.5_

- [ ]* 14.2 Write property test for jump navigation determinism
  - **Property 10: Jump Navigation Determinism**
  - **Validates: Requirements 8.2, 8.3, 22.1, 22.2, 22.3, 22.4, 22.5**

- [ ] 15. Find Within Video Service
- [ ] 15.1 Implement FindWithinVideoService
  - find_next() method
  - find_prev() method
  - _search_transcript_fts() using PostgreSQL full-text search
  - _search_ocr_fts() using PostgreSQL full-text search
  - Support multi-source search (transcript, ocr, all)
  - _Requirements: 9.2, 9.3, 23.1, 23.2, 23.3_

- [ ]* 15.2 Write property test for find direction ordering
  - **Property 11: Find Direction Ordering**
  - **Validates: Requirements 9.2, 9.3, 23.2, 23.3**

- [ ]* 15.3 Write property test for multi-source find merging
  - **Property 12: Multi-Source Find Merging**
  - **Validates: Requirements 23.1, 23.3, 23.4, 23.5**

- [ ] 16. API Endpoints
- [ ] 16.1 Implement GET /v1/videos/{video_id}/jump endpoint
  - Support all artifact types via kind parameter
  - Support label, face_cluster_id, min_confidence filtering
  - Support selection and profile parameters
  - Return 404 when no match found
  - _Requirements: 8.1, 8.4, 8.5, 22.1_

- [ ] 16.2 Implement GET /v1/videos/{video_id}/find endpoint
  - Support query parameter
  - Support direction (next/prev)
  - Support source (transcript, ocr, all)
  - Return matches with snippets
  - _Requirements: 9.1, 9.4, 9.5, 23.4_

- [ ] 16.3 Implement GET /v1/videos/{video_id}/artifacts endpoint
  - Support type filtering
  - Support time range filtering (from_ms, to_ms)
  - Support selection and profile parameters
  - Return artifacts with payloads
  - _Requirements: 24.1, 24.5_

- [ ]* 16.4 Write integration tests for API endpoints
  - Test jump endpoint for all artifact types
  - Test find endpoint with different sources
  - Test artifacts query endpoint
  - Test error cases (404, 400)

- [ ] 17. Multi-Profile Support
- [ ] 17.1 Update task handlers to include model_profile in artifact creation
  - Pass model_profile from worker configuration
  - _Requirements: 11.1_

- [ ] 17.2 Implement GET /v1/videos/{video_id}/profiles endpoint
  - List available profiles for asset and artifact_type
  - _Requirements: 11.5_

- [ ]* 17.3 Write property test for multi-profile preservation
  - **Property 14: Multi-Profile Preservation**
  - **Validates: Requirements 11.1, 11.4**

- [x] 18. Legacy Data Access
- [x] 18.1 Add legacy flag to existing API responses
  - Modify legacy endpoints to indicate data source
  - _Requirements: 10.3_

- [x] 18.2 Document which videos have artifact vs legacy data
  - Add metadata field to video model
  - _Requirements: 10.5_

- [ ]* 18.3 Write property test for legacy data access
  - **Property 13: Legacy Data Access**
  - **Validates: Requirements 10.1, 10.2, 10.3**

- [x] 19. Worker Pool Session Isolation and Batch Operations (Bug Fix)
- [x] 19.1 Add batch_create method to ArtifactRepository
  - Implement batch_create(artifacts: list[ArtifactEnvelope]) method
  - Validate all artifacts before inserting any
  - Use single transaction for all inserts
  - Rollback entire batch on any validation error
  - _Requirements: 7.1 (Repository CRUD operations)_

- [x] 19.2 Update worker pool to use per-worker session instances
  - Move session creation inside worker loop (not shared across workers)
  - Create new session at start of each task processing
  - Ensure session is closed after each task completes
  - Add proper exception handling with session rollback
  - _Requirements: 2.1 (Run tracking and error handling)_

- [x] 19.3 Update task handlers to use batch artifact creation
  - Modify WhisperTranscriptionService to collect all segments before insert
  - Modify ObjectDetectionService to batch all detections
  - Modify FaceDetectionService to batch all detections
  - Modify PlaceClassificationService to batch all classifications
  - Modify OcrService to batch all text detections
  - Use repository.batch_create() instead of individual creates
  - _Requirements: 4.2, 13.2, 14.2, 15.2, 16.2, 17.2_

- [ ]* 19.4 Write integration test for worker isolation
  - Test that one worker's session failure doesn't affect other workers
  - Simulate artifact validation error in one worker
  - Verify other workers continue processing successfully
  - _Requirements: 2.4 (Run failure handling)_

- [ ]* 19.5 Write unit test for batch artifact creation
  - Test successful batch insert of multiple artifacts
  - Test rollback when one artifact fails validation
  - Test that no artifacts are inserted if any fail
  - _Requirements: 7.1, 1.2 (Repository operations and validation)_

- [x] 20. Final Integration and Testing
- [x] 20.1 Update worker pool manager to use artifact-based services
  - Wire up all new services
  - Remove legacy service calls

- [x] 20.2 Run full integration test suite
  - Process test video through all pipelines
  - Verify artifacts created correctly
  - Verify projections synchronized
  - Verify API endpoints return correct data

- [x] 20.3 Performance testing
  - Test query performance on large artifact sets
  - Verify indexes are being used
  - Monitor database size growth

- [ ] 21. Documentation
- [ ] 21.1 Update API documentation
  - Document new endpoints
  - Document selection policy usage
  - Document multi-profile support

- [ ] 21.2 Create developer guide for adding new artifact types
  - Schema registration process
  - Projection creation
  - Service integration

- [ ] 22. Final Checkpoint
- Ensure all tests pass, verify performance meets requirements, ask the user if ready for deployment.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end flows
