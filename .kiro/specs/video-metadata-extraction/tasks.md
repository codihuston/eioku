# Implementation Plan: Video Metadata Extraction

## Overview

This implementation plan breaks down the video metadata extraction feature into incremental, testable tasks. The feature integrates metadata extraction into the video discovery pipeline, persists results as artifact envelopes, normalizes GPS data to a projection table, and displays metadata in the video player UI.

## Tasks

- [x] 1. Create metadata artifact schema and register with SchemaRegistry
  - Create MetadataV1 schema class with all metadata fields
  - Register "video.metadata" artifact type with SchemaRegistry
  - Add schema validation tests
  - _Requirements: 2.1, 2.3, 2.6_

- [ ]* 1.1 Write property test for metadata artifact schema
  - **Property 2: Metadata Artifact Envelope Structure**
  - **Validates: Requirements 2.1, 2.2, 2.4, 2.5**

- [x] 2. Create video_locations projection table
  - Create PostgreSQL migration for video_locations table
  - Add indexes on asset_id, latitude, longitude
  - Add foreign key constraints to artifacts and videos tables
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 2.5 Add file_created_at field to videos table
  - Create PostgreSQL migration to add file_created_at column to videos table
  - Make column nullable (for videos without metadata)
  - Add index on file_created_at for efficient "next/prev video" queries
  - Populate from EXIF create_date (preferred), file system mtime (fallback), or discovery timestamp
  - _Requirements: Future "next video" navigation feature_

- [x] 3. Implement MetadataExtractor in ML service
  - Create MetadataExtractor class using pyexiftool
  - Extract Composite fields with QuickTime fallbacks
  - Omit null values from output
  - Handle extraction errors gracefully
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 6.1, 6.2_

- [ ]* 3.1 Write property test for metadata extraction
  - **Property 6: Metadata Extraction Handles Missing Fields**
  - **Validates: Requirements 1.4, 1.5, 6.1, 6.2**

- [x] 4. Add metadata_extraction task type to task registry
  - Define metadata_extraction in TASK_REGISTRY
  - Mark as language-agnostic (language not required)
  - Set default configuration in content_creator.json
  - _Requirements: 4.1, 4.3_

- [x] 5. Update VideoDiscoveryService to create metadata tasks
  - Add "metadata_extraction" to ACTIVE_TASK_TYPES
  - Create metadata_extraction task with language=NULL on discovery
  - Enqueue task to job queue
  - _Requirements: 4.1, 4.2, 4.4_

- [ ]* 5.1 Write property test for metadata task creation
  - **Property 1: Metadata Task Creation on Discovery**
  - **Validates: Requirements 4.1, 4.2, 4.4**

- [~] 6. Create ArtifactTransformer mapping for metadata_extraction
  - Add metadata_extraction to ARTIFACT_SCHEMA_MAP
  - Map to "video.metadata" artifact type
  - Set producer="pyexiftool", model_profile="balanced"
  - _Requirements: 2.1, 2.4, 2.5_

- [ ] 7. Implement metadata artifact persistence
  - Create artifact envelope from extraction results
  - Set span_start_ms=0, span_end_ms=video_duration_ms
  - Persist to artifacts table with schema_version=1
  - _Requirements: 2.1, 2.2, 2.5, 2.6_

- [ ]* 7.1 Write property test for artifact persistence
  - **Property 9: Metadata Artifact Persistence**
  - **Validates: Requirements 2.6**

- [ ] 8. Implement GPS projection sync in ProjectionSyncService
  - Add _sync_video_metadata method to ProjectionSyncService
  - Extract GPS coordinates from metadata artifact payload
  - Create video_locations entry if GPS exists
  - Handle missing GPS gracefully
  - _Requirements: 3.1, 3.2, 3.4, 3.5, 6.4_

- [ ]* 8.1 Write property test for GPS projection creation
  - **Property 4: GPS Projection Table Creation**
  - **Validates: Requirements 3.1, 3.2, 3.5**

- [ ]* 8.2 Write property test for no GPS projection without coordinates
  - **Property 5: No GPS Projection Without Coordinates**
  - **Validates: Requirements 3.4**

- [ ] 9. Checkpoint - Ensure all backend tests pass
  - Run all backend tests: `cd backend && poetry run pytest tests/`
  - Verify metadata extraction, artifact persistence, and projection sync
  - Ask the user if questions arise.

- [ ] 10. Create MetadataViewer React component
  - Create component to display metadata in organized sections
  - Render GPS coordinates in user-friendly format
  - Render Camera, File, Temporal, and Image sections
  - Handle missing fields gracefully
  - _Requirements: 5.2, 5.3, 5.4, 5.5_

- [ ]* 10.1 Write property test for MetadataViewer rendering
  - **Property 7: Metadata Viewer Displays Available Fields**
  - **Validates: Requirements 5.2, 5.3, 5.5**

- [ ]* 10.2 Write property test for GPS coordinate formatting
  - **Property 8: GPS Coordinate Formatting**
  - **Validates: Requirements 5.4**

- [ ] 11. Add Metadata tab to VideoPlayer component
  - Add "Metadata" tab button alongside existing tabs
  - Implement tab switching logic
  - Render MetadataViewer when Metadata tab is active
  - _Requirements: 5.1_

- [ ]* 11.1 Write unit test for VideoPlayer Metadata tab
  - Test that Metadata tab is rendered
  - Test that tab switching works
  - _Requirements: 5.1_

- [ ] 12. Implement metadata fetching in frontend
  - Create hook to fetch metadata artifacts from API
  - Handle loading and error states
  - Cache metadata artifacts
  - _Requirements: 5.2_

- [ ] 13. Checkpoint - Ensure all frontend tests pass
  - Run all frontend tests: `cd frontend && npm run test`
  - Verify MetadataViewer and VideoPlayer components
  - Ask the user if questions arise.

- [ ] 14. Integration test - End-to-end metadata extraction
  - Create test video with metadata
  - Discover video and verify metadata_extraction task created
  - Verify artifact persisted to database
  - Verify video_locations entry created if GPS exists
  - Verify metadata displayed in UI
  - _Requirements: 1.1, 2.1, 3.1, 5.2_

- [ ] 15. Integration test - Error handling
  - Test with unreadable video file
  - Test with video lacking metadata
  - Test with video lacking GPS coordinates
  - Verify graceful error handling
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 16. Final checkpoint - Ensure all tests pass
  - Run full test suite: backend and frontend
  - Verify no regressions in existing functionality
  - Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows

