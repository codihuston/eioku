# Requirements Document

## Introduction

This document specifies Phase 1 of the LEAP PLAYER artifact envelope architecture migration. The system will transition from a "one type = one table" pattern to a unified artifact envelope model that provides canonical, time-aligned artifacts for deterministic navigation while enabling multiple model strengths to coexist. Phase 1 focuses on establishing the core artifact infrastructure and migrating all video processing tasks (transcripts, scenes, objects, faces, places, OCR text) to the unified model. Video hash computation remains in its current implementation as it serves a different purpose (content identification rather than time-aligned analysis).

## Glossary

- **Artifact**: A canonical record of ML output or metadata with a time span, stored in a unified envelope format
- **Artifact_Envelope**: The shared structure containing artifact_id, asset_id, artifact_type, span, payload, producer metadata, and versioning information
- **Asset**: A video file with associated metadata (formerly referred to as "video" in the codebase)
- **Payload**: The type-specific JSON data stored within an artifact envelope
- **Producer**: The tool or model that generated an artifact (e.g., "whisper", "pyscenedetect")
- **Model_Profile**: The strength/quality tier of a model run (fast, balanced, high_quality)
- **Run**: A pipeline execution that produces artifacts for an asset
- **Projection**: A derived table that denormalizes artifact data for query performance
- **Schema_Version**: An integer version number for artifact payload schemas
- **Selection_Policy**: Rules determining which artifact version to use by default (latest, pinned, profile-based)
- **Transcript_Segment**: A time-aligned piece of transcribed text with speaker and confidence information
- **Scene**: A continuous video segment detected by scene change detection algorithms
- **Object_Detection**: A detected object in video frames with bounding box, label, and confidence
- **Face_Detection**: A detected face in video frames with bounding box and optional cluster assignment
- **Place_Classification**: A scene/location classification (e.g., "kitchen", "office", "beach")
- **OCR_Text**: Text detected and extracted from video frames
- **FTS5**: SQLite Full-Text Search version 5, used for keyword search

## Requirements

### Requirement 1: Artifact Envelope Data Model

**User Story:** As a system architect, I want a unified artifact storage model, so that adding new ML capabilities doesn't require new tables and services.

#### Acceptance Criteria

1. THE System SHALL store artifacts in a unified `artifacts` table with envelope fields: artifact_id, asset_id, artifact_type, schema_version, span_start_ms, span_end_ms, payload_json, producer, producer_version, model_profile, config_hash, input_hash, run_id, created_at
2. THE System SHALL validate artifact payloads against registered schemas based on (artifact_type, schema_version)
3. THE System SHALL support multiple schema versions for the same artifact_type simultaneously
4. THE System SHALL index artifacts by (asset_id, artifact_type, span_start_ms) for efficient timeline queries
5. THE System SHALL index artifacts by (asset_id, artifact_type, model_profile, span_start_ms) for profile-filtered queries

### Requirement 2: Run Tracking

**User Story:** As a developer, I want to track pipeline executions, so that I can debug issues and understand artifact provenance.

#### Acceptance Criteria

1. THE System SHALL store pipeline runs in a `runs` table with fields: run_id, asset_id, pipeline_profile, started_at, finished_at, status, error
2. WHEN a pipeline execution starts, THE System SHALL create a run record with status "running"
3. WHEN a pipeline execution completes, THE System SHALL update the run record with status "completed" and finished_at timestamp
4. IF a pipeline execution fails, THEN THE System SHALL update the run record with status "failed" and error message
5. THE System SHALL link artifacts to their originating run via run_id

### Requirement 3: Artifact Schema Registry

**User Story:** As a developer, I want centralized schema definitions, so that artifact validation is consistent and type-safe.

#### Acceptance Criteria

1. THE System SHALL maintain a registry mapping (artifact_type, schema_version) to Pydantic models
2. THE System SHALL validate artifact payloads on write using the registered schema
3. THE System SHALL deserialize artifact payloads on read using the registered schema
4. WHEN an unregistered (artifact_type, schema_version) is encountered, THE System SHALL raise a SchemaNotFoundError
5. THE System SHALL support registering new schemas without modifying the artifacts table

### Requirement 4: Transcript Artifact Storage

**User Story:** As a system operator, I want transcript data stored in the artifact model, so that transcripts benefit from versioning and multi-model support.

#### Acceptance Criteria

1. THE System SHALL define a "transcript.segment" artifact type with schema version 1
2. THE System SHALL store transcript segments as artifacts with payload containing: text, speaker, confidence, language
3. WHEN processing a video with Whisper, THE System SHALL create transcript.segment artifacts
4. THE System SHALL create corresponding entries in transcript_fts projection for full-text search
5. THE System SHALL preserve all metadata (timestamps, confidence, speaker) in artifact payloads

### Requirement 5: Transcript FTS Projection

**User Story:** As a user, I want fast keyword search in transcripts, so that I can find specific phrases quickly.

#### Acceptance Criteria

1. THE System SHALL create a `transcript_fts` FTS5 table with columns: artifact_id, asset_id, start_ms, end_ms, text
2. WHEN a transcript.segment artifact is created, THE System SHALL insert a corresponding row into transcript_fts
3. THE System SHALL support FTS5 MATCH queries for keyword search across transcript text
4. THE System SHALL return search results with artifact_id, time span, and highlighted snippets
5. THE System SHALL index transcript_fts by asset_id for per-video search performance

### Requirement 6: Selection Policy

**User Story:** As a user, I want to choose which model version to use, so that I can balance speed vs quality based on my needs.

#### Acceptance Criteria

1. THE System SHALL store selection policies in an `artifact_selections` table with fields: asset_id, artifact_type, selection_mode, preferred_profile, pinned_run_id, pinned_artifact_id, updated_at
2. THE System SHALL support selection modes: "default", "pinned", "latest", "profile", "best_quality"
3. WHEN selection_mode is "profile" and preferred_profile is set, THE System SHALL return artifacts matching that profile
4. WHEN selection_mode is "pinned" and pinned_run_id is set, THE System SHALL return artifacts from that specific run
5. WHEN no selection policy exists, THE System SHALL default to returning the most recent artifacts

### Requirement 7: Artifact Repository

**User Story:** As a developer, I want a clean repository interface for artifacts, so that I can query and store artifacts without SQL boilerplate.

#### Acceptance Criteria

1. THE System SHALL provide an ArtifactRepository with methods: create, get_by_id, get_by_asset, get_by_span, delete
2. THE System SHALL provide a get_by_asset method that filters by asset_id, artifact_type, and optional time range
3. THE System SHALL provide a get_by_span method that returns artifacts overlapping a given time range
4. THE System SHALL apply selection policies when querying artifacts through the repository
5. THE System SHALL handle schema validation and serialization transparently in the repository layer

### Requirement 8: Jump Navigation API

**User Story:** As a user, I want to jump to the next transcript segment, so that I can navigate through spoken content efficiently.

#### Acceptance Criteria

1. THE System SHALL provide a GET /v1/videos/{video_id}/jump endpoint with parameters: kind, direction, from_ms, selection, profile
2. WHEN kind is "transcript" and direction is "next", THE System SHALL return the next transcript segment after from_ms
3. WHEN kind is "transcript" and direction is "prev", THE System SHALL return the previous transcript segment before from_ms
4. THE System SHALL return jump targets with fields: jump_to (start_ms, end_ms), artifact_ids
5. IF no matching artifact exists, THEN THE System SHALL return a 404 response with an appropriate message

### Requirement 9: Find Within Video API

**User Story:** As a user, I want to search for keywords within a video, so that I can jump to relevant moments.

#### Acceptance Criteria

1. THE System SHALL provide a GET /v1/videos/{video_id}/find endpoint with parameters: q (query), direction, from_ms, selection, profile
2. WHEN direction is "next", THE System SHALL return transcript matches after from_ms ordered by time ascending
3. WHEN direction is "prev", THE System SHALL return transcript matches before from_ms ordered by time descending
4. THE System SHALL return matches with fields: jump_to (start_ms, end_ms), artifact_id, snippet (highlighted text)
5. THE System SHALL use FTS5 snippet function to generate highlighted text excerpts

### Requirement 10: Historical Data Access

**User Story:** As a user, I want to access historical video processing results, so that I can reference previously processed videos.

#### Acceptance Criteria

1. THE System SHALL maintain legacy tables as read-only for historical data reference
2. THE System SHALL provide API endpoints to query legacy data when needed
3. THE System SHALL clearly distinguish between artifact-based (new) and legacy (historical) data in responses
4. THE System SHALL allow optional reprocessing of old videos to populate artifacts
5. THE System SHALL document which videos have artifact data vs legacy-only data

### Requirement 11: Multiple Model Strengths

**User Story:** As a user, I want to run different model strengths on the same video, so that I can choose between fast preview and high-quality results.

#### Acceptance Criteria

1. THE System SHALL store artifacts from different model_profiles (fast, balanced, high_quality) without overwriting each other
2. THE System SHALL allow querying artifacts filtered by model_profile
3. WHEN multiple profiles exist for the same artifact_type, THE System SHALL return the profile specified in the selection policy
4. THE System SHALL preserve all historical artifacts even when new runs are executed
5. THE System SHALL provide an API to list available profiles for a given asset and artifact_type

### Requirement 12: Artifact Metadata and Provenance

**User Story:** As a developer, I want complete provenance information for artifacts, so that I can reproduce results and debug issues.

#### Acceptance Criteria

1. THE System SHALL store producer name and version for every artifact
2. THE System SHALL store config_hash representing the configuration used to generate the artifact
3. THE System SHALL store input_hash representing the input data fingerprint
4. THE System SHALL allow querying artifacts by producer, producer_version, or config_hash
5. THE System SHALL provide an API endpoint to retrieve full provenance information for an artifact_id

### Requirement 13: Scene Detection Artifact Storage

**User Story:** As a system operator, I want scene detection data stored in the artifact model, so that scenes benefit from versioning and multi-model support.

#### Acceptance Criteria

1. THE System SHALL define a "scene" artifact type with schema version 1
2. THE System SHALL store scene detections as artifacts with payload containing: scene_index, method, score, frame_number
3. WHEN processing a video with scene detection, THE System SHALL create scene artifacts
4. THE System SHALL create corresponding entries in scene_ranges projection for fast navigation
5. THE System SHALL preserve scene ordering and timestamps in artifact payloads

### Requirement 14: Object Detection Artifact Storage

**User Story:** As a system operator, I want object detection data stored in the artifact model, so that object detections benefit from versioning and multi-model support.

#### Acceptance Criteria

1. THE System SHALL define an "object.detection" artifact type with schema version 1
2. THE System SHALL store object detections as artifacts with payload containing: label, confidence, bounding_box (x, y, width, height), frame_number
3. WHEN processing a video with object detection, THE System SHALL create object.detection artifacts
4. THE System SHALL create corresponding entries in object_labels projection for fast label-based queries
5. THE System SHALL create one artifact per object occurrence with frame-level granularity

### Requirement 15: Face Detection Artifact Storage

**User Story:** As a system operator, I want face detection data stored in the artifact model, so that face detections benefit from versioning and multi-model support.

#### Acceptance Criteria

1. THE System SHALL define a "face.detection" artifact type with schema version 1
2. THE System SHALL store face detections as artifacts with payload containing: confidence, bounding_box (x, y, width, height), cluster_id, frame_number
3. WHEN processing a video with face detection, THE System SHALL create face.detection artifacts
4. THE System SHALL create corresponding entries in face_clusters projection for fast cluster-based queries
5. THE System SHALL preserve face cluster assignments in artifact payloads

#### Acceptance Criteria

1. THE System SHALL define a "face.detection" artifact type with schema version 1
2. THE System SHALL store face detections as artifacts with payload containing: confidence, bounding_box (x, y, width, height), cluster_id, frame_number
3. WHEN processing a video with face detection, THE System SHALL create face.detection artifacts
4. THE System SHALL provide a migration script to convert existing face records to artifacts
5. THE System SHALL preserve face cluster assignments during migration

### Requirement 16: Place Classification Artifact Storage

**User Story:** As a system operator, I want place classification data stored in the artifact model, so that place detections benefit from versioning and multi-model support.

#### Acceptance Criteria

1. THE System SHALL define a "place.classification" artifact type with schema version 1
2. THE System SHALL store place classifications as artifacts with payload containing: label, confidence, alternative_labels (list of label-confidence pairs), frame_number
3. WHEN processing a video with place detection, THE System SHALL create place.classification artifacts
4. THE System SHALL preserve alternative labels and confidence scores in artifact payloads
5. THE System SHALL support querying places by label through artifact queries

### Requirement 17: OCR Text Detection Artifact Storage

**User Story:** As a system operator, I want OCR text detection data stored in the artifact model, so that OCR results benefit from versioning and multi-model support.

#### Acceptance Criteria

1. THE System SHALL define an "ocr.text" artifact type with schema version 1
2. THE System SHALL store OCR detections as artifacts with payload containing: text, confidence, bounding_box (polygon points), language, frame_number
3. WHEN processing a video with OCR detection, THE System SHALL create ocr.text artifacts
4. THE System SHALL create corresponding entries in ocr_fts projection for full-text search
5. THE System SHALL preserve bounding box polygons and language metadata in artifact payloads

### Requirement 18: Object Label Projection

**User Story:** As a user, I want fast "find next object" queries, so that I can jump to the next occurrence of a specific object label.

#### Acceptance Criteria

1. THE System SHALL create an `object_labels` projection table with columns: artifact_id, asset_id, label, confidence, start_ms, end_ms
2. WHEN an object.detection artifact is created, THE System SHALL insert a corresponding row into object_labels
3. THE System SHALL index object_labels by (asset_id, label, start_ms) for fast label-filtered queries
4. THE System SHALL support querying next/previous object occurrence by label
5. THE System SHALL support minimum confidence filtering in object label queries

### Requirement 19: Face Cluster Projection

**User Story:** As a user, I want fast "find next face" queries by cluster, so that I can navigate through appearances of specific people.

#### Acceptance Criteria

1. THE System SHALL create a `face_clusters` projection table with columns: artifact_id, asset_id, cluster_id, confidence, start_ms, end_ms
2. WHEN a face.detection artifact with cluster_id is created, THE System SHALL insert a corresponding row into face_clusters
3. THE System SHALL index face_clusters by (asset_id, cluster_id, start_ms) for fast cluster-filtered queries
4. THE System SHALL support querying next/previous face occurrence by cluster_id
5. THE System SHALL support minimum confidence filtering in face cluster queries

### Requirement 20: Scene Range Projection

**User Story:** As a user, I want fast scene navigation, so that I can jump between scenes efficiently.

#### Acceptance Criteria

1. THE System SHALL create a `scene_ranges` projection table with columns: artifact_id, asset_id, scene_index, start_ms, end_ms
2. WHEN a scene artifact is created, THE System SHALL insert a corresponding row into scene_ranges
3. THE System SHALL index scene_ranges by (asset_id, scene_index) for fast scene navigation
4. THE System SHALL support querying next/previous scene by scene_index
5. THE System SHALL maintain scene_index ordering for deterministic navigation

### Requirement 21: OCR Text FTS Projection

**User Story:** As a user, I want fast keyword search in OCR text, so that I can find specific on-screen text quickly.

#### Acceptance Criteria

1. THE System SHALL create an `ocr_fts` FTS5 table with columns: artifact_id, asset_id, start_ms, end_ms, text
2. WHEN an ocr.text artifact is created, THE System SHALL insert a corresponding row into ocr_fts
3. THE System SHALL support FTS5 MATCH queries for keyword search across OCR text
4. THE System SHALL return search results with artifact_id, time span, and highlighted snippets
5. THE System SHALL index ocr_fts by asset_id for per-video search performance

### Requirement 22: Extended Jump Navigation API

**User Story:** As a user, I want to jump to different artifact types (scenes, objects, faces, places, OCR), so that I can navigate through various video features.

#### Acceptance Criteria

1. THE System SHALL extend the GET /v1/videos/{video_id}/jump endpoint to support kind values: "scene", "transcript", "object", "face", "place", "ocr"
2. WHEN kind is "object", THE System SHALL support a label parameter to filter by object label
3. WHEN kind is "face", THE System SHALL support a face_cluster_id parameter to filter by face cluster
4. WHEN kind is "place", THE System SHALL support a label parameter to filter by place label
5. WHEN kind is "ocr", THE System SHALL support a text parameter for OCR text matching

### Requirement 23: Extended Find Within Video API

**User Story:** As a user, I want to search for keywords in both transcripts and OCR text, so that I can find spoken or on-screen content.

#### Acceptance Criteria

1. THE System SHALL extend the GET /v1/videos/{video_id}/find endpoint to support searching both transcript and OCR text
2. THE System SHALL support a source parameter with values: "transcript", "ocr", "all" (default)
3. WHEN source is "all", THE System SHALL search both transcript_fts and ocr_fts and merge results by timestamp
4. THE System SHALL return matches with a source field indicating whether the match came from transcript or OCR
5. THE System SHALL maintain timestamp ordering when merging results from multiple sources

### Requirement 24: API Consistency

**User Story:** As an API consumer, I want consistent artifact-based endpoints, so that I can access all video processing results through a unified interface.

#### Acceptance Criteria

1. THE System SHALL provide GET /v1/videos/{video_id}/artifacts endpoint for querying artifacts by type
2. THE System SHALL provide GET /v1/videos/{video_id}/jump endpoint for navigation across all artifact types
3. THE System SHALL provide GET /v1/videos/{video_id}/find endpoint for keyword search
4. THE System SHALL return consistent response formats across all artifact types
5. THE System SHALL support filtering by model_profile and selection policies in all endpoints
