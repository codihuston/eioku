# Requirements Document: Worker & ML Service Separation

## Introduction

This document specifies the requirements for separating the monolithic Eioku backend into three independent services: API Service, Worker Service, and ML Service. This separation enables independent scaling, better resource isolation (CPU vs GPU), and cleaner deployment boundaries while maintaining all existing functionality.

## Glossary

- **API Service**: FastAPI server handling REST endpoints, business logic, and database management
- **Worker Service**: arq-based job consumer that orchestrates ML task execution
- **ML Service**: Stateless FastAPI server exposing HTTP endpoints for ML inference operations
- **Task**: A unit of work representing one ML operation on a video (e.g., object detection, transcription)
- **Job**: A Redis-backed arq job representing a task's execution in the queue
- **Artifact**: Result data from ML inference (detections, transcriptions, etc.) stored in PostgreSQL
- **Reconciler**: Background process that synchronizes PostgreSQL task state with Redis job state
- **Provenance**: Metadata about ML inference including config_hash and input_hash for reproducibility
- **arq**: Python async job queue library using Redis as message broker
- **Valkey**: Linux Foundation fork of Redis, 100% protocol-compatible
- **GPU Semaphore**: Concurrency control mechanism limiting simultaneous GPU operations
- **Job Abort**: Capability to cancel a queued or running job via API

## Requirements

### Requirement 1: API Service Functionality

**User Story:** As a system operator, I want the API Service to maintain all existing REST endpoints and orchestrate task creation, so that the system remains backward compatible and functional.

#### Acceptance Criteria

1. THE API_Service SHALL maintain all existing REST endpoints (videos, tasks, artifacts, paths, selections)
2. WHEN a video is discovered, THE API_Service SHALL create task records in PostgreSQL
3. WHEN a task is created, THE API_Service SHALL enqueue a job to Redis via arq
4. WHEN a client requests task status, THE API_Service SHALL return current status from PostgreSQL including job_id and timestamps
5. THE API_Service SHALL NOT execute ML inference operations directly (delegated to ML Service)
6. THE API_Service SHALL NOT consume jobs from Redis (delegated to Worker Service)
7. WHEN the API_Service starts, THE API_Service SHALL establish connection pool to PostgreSQL (max_size=20)
8. WHEN the API_Service starts, THE API_Service SHALL establish connection to Redis for job enqueueing

### Requirement 2: Worker Service Job Consumption

**User Story:** As a system operator, I want the Worker Service to consume jobs from Redis and dispatch them to the ML Service, so that ML tasks execute asynchronously and independently from the API.

#### Acceptance Criteria

1. WHEN the Worker_Service starts, THE Worker_Service SHALL read GPU_MODE environment variable (gpu, cpu, or auto)
2. WHEN GPU_MODE is set to gpu, THE Worker_Service SHALL only consume jobs from the gpu_jobs queue
3. WHEN GPU_MODE is set to cpu, THE Worker_Service SHALL only consume jobs from the cpu_jobs queue
4. WHEN GPU_MODE is set to auto, THE Worker_Service SHALL detect GPU availability and consume from appropriate queue
5. WHEN the Worker_Service starts, THE Worker_Service SHALL connect to Redis and begin consuming jobs from the appropriate queue
6. WHEN a job is consumed, THE Worker_Service SHALL update the corresponding task status to RUNNING in PostgreSQL
7. WHEN a job is consumed, THE Worker_Service SHALL dispatch an HTTP request to the ML_Service with task parameters
8. WHEN the ML_Service returns results, THE Worker_Service SHALL batch insert artifacts into PostgreSQL
9. WHEN artifacts are inserted, THE Worker_Service SHALL update the task status to COMPLETED in PostgreSQL
10. WHEN the Worker_Service completes a job, THE Worker_Service SHALL acknowledge the job in Redis (XACK)
11. THE Worker_Service SHALL support configurable concurrency (max_jobs parameter)
12. THE Worker_Service SHALL NOT expose HTTP endpoints (no REST API)
13. WHEN the Worker_Service starts, THE Worker_Service SHALL establish connection pool to PostgreSQL (max_size=10)

### Requirement 3: ML Service Inference Endpoints

**User Story:** As a Worker Service, I want to call stateless HTTP endpoints for ML operations, so that I can dispatch inference tasks without managing ML models locally.

#### Acceptance Criteria

1. THE ML_Service SHALL expose POST /infer/objects endpoint for object detection
2. THE ML_Service SHALL expose POST /infer/faces endpoint for face detection
3. THE ML_Service SHALL expose POST /infer/transcribe endpoint for transcription
4. THE ML_Service SHALL expose POST /infer/ocr endpoint for optical character recognition
5. THE ML_Service SHALL expose POST /infer/places endpoint for place classification
6. THE ML_Service SHALL expose POST /infer/scenes endpoint for scene detection
7. THE ML_Service SHALL expose GET /health endpoint returning model status and GPU availability
8. WHEN an inference endpoint receives a request, THE ML_Service SHALL return structured results including detections/classifications and provenance metadata
9. WHEN an inference endpoint receives a request, THE ML_Service SHALL include config_hash and input_hash in the response for reproducibility
10. THE ML_Service SHALL NOT access PostgreSQL directly
11. THE ML_Service SHALL NOT consume jobs from Redis
12. WHEN the ML_Service starts, THE ML_Service SHALL lazy-load ML models on first request to that endpoint

### Requirement 4: Job Queue & Redis Integration

**User Story:** As a system operator, I want jobs to flow through Redis with arq, so that tasks can be reliably queued and processed asynchronously.

#### Acceptance Criteria

1. WHEN a task is created in the API_Service, THE API_Service SHALL enqueue a job to Redis with task_id, task_type, video_id, and video_path
2. WHEN a job is enqueued, THE API_Service SHALL determine if the task requires GPU or can run on CPU
3. WHEN a job requires GPU, THE API_Service SHALL enqueue it to the gpu_jobs queue
4. WHEN a job can run on CPU, THE API_Service SHALL enqueue it to the cpu_jobs queue
5. WHEN a job is enqueued, THE API_Service SHALL assign a unique job_id in format "ml_{task_id}" for deduplication
6. WHEN a job is enqueued, THE API_Service SHALL set _queue_name to either gpu_jobs or cpu_jobs based on task type
7. WHEN the Worker_Service consumes a job, THE Worker_Service SHALL use arq's XREADGROUP to read from the appropriate consumer group
8. WHEN a job completes successfully, THE Worker_Service SHALL acknowledge it via XACK
9. WHEN a job fails, THE Worker_Service SHALL allow arq to retry with exponential backoff (max_tries=3)
10. THE Redis instance SHALL use Valkey (Linux Foundation fork) for persistence and reliability
11. WHEN Redis is configured, THE system SHALL enable AOF (Append-Only File) persistence for durability
12. THE system SHALL maintain separate consumer groups for gpu_jobs and cpu_jobs queues

### Requirement 5: Task Status Synchronization

**User Story:** As a system operator, I want task status to be synchronized between PostgreSQL and Redis, so that the system maintains consistency and can recover from failures.

#### Acceptance Criteria

1. WHEN a task is created, THE task status SHALL be PENDING in PostgreSQL
2. WHEN a Worker_Service consumes a job, THE task status SHALL transition to RUNNING
3. WHEN a Worker_Service completes a job, THE task status SHALL transition to COMPLETED
4. WHEN a Worker_Service fails a job, THE task status SHALL transition to FAILED with error message
5. WHEN a task is cancelled via API, THE task status SHALL transition to CANCELLED
6. WHEN a task transitions to RUNNING, THE started_at timestamp SHALL be recorded
7. WHEN a task transitions to COMPLETED or FAILED, THE completed_at timestamp SHALL be recorded
8. THE task record SHALL include error field for storing failure reasons

### Requirement 6: Reconciliation & Failure Recovery

**User Story:** As a system operator, I want the system to automatically recover from failures without losing jobs, so that the system is resilient to service restarts and Redis data loss.

#### Acceptance Criteria

1. WHEN the Reconciler runs, THE Reconciler SHALL check all PENDING tasks in PostgreSQL
2. IF a PENDING task has no corresponding job in Redis, THEN THE Reconciler SHALL enqueue the job (handles Redis data loss)
3. WHEN the Reconciler runs, THE Reconciler SHALL check all RUNNING tasks in PostgreSQL
4. IF a RUNNING task has no corresponding job in Redis, THEN THE Reconciler SHALL reset status to PENDING and re-enqueue (handles job loss)
5. IF a RUNNING task's job shows status=complete in Redis, THEN THE Reconciler SHALL update PostgreSQL to COMPLETED
6. IF a RUNNING task's job shows status=failed in Redis, THEN THE Reconciler SHALL update PostgreSQL to FAILED with error message
7. THE Reconciler SHALL run periodically (every 5 minutes)
8. THE Reconciler SHALL NOT use time-based thresholds to kill jobs (only acts on definitive Redis signals)
9. WHEN a task exceeds alert threshold, THE Reconciler SHALL send alert to operator (not auto-kill)
10. THE Reconciler SHALL use PostgreSQL as source of truth for task state

### Requirement 7: GPU Resource Management

**User Story:** As a system operator, I want GPU resources to be managed efficiently, so that concurrent ML inference doesn't overwhelm GPU memory.

#### Acceptance Criteria

1. WHEN the ML_Service starts, THE ML_Service SHALL initialize a GPU semaphore with configurable concurrency limit
2. WHEN an inference endpoint receives a request, THE ML_Service SHALL acquire the GPU semaphore before loading models
3. WHEN an inference endpoint completes, THE ML_Service SHALL release the GPU semaphore
4. THE GPU semaphore limit SHALL be configurable via environment variable (default=2)
5. WHEN GPU semaphore is at capacity, THE ML_Service SHALL queue requests until a slot becomes available
6. THE ML_Service SHALL support lazy model loading (load on first request, not at startup)

### Requirement 8: Artifact Persistence & Provenance

**User Story:** As a system operator, I want ML artifacts to be persisted with provenance metadata, so that results are reproducible and traceable.

#### Acceptance Criteria

1. WHEN the Worker_Service receives inference results from ML_Service, THE Worker_Service SHALL extract config_hash and input_hash
2. WHEN the Worker_Service persists artifacts, THE Worker_Service SHALL batch insert all artifacts for a task in a single transaction
3. WHEN artifacts are inserted, THE artifacts table SHALL include config_hash and input_hash columns
4. WHEN artifacts are inserted, THE artifacts table SHALL include producer and model_profile columns
5. WHEN artifacts are inserted, THE artifacts table SHALL include run_id for linking to execution context
6. THE Worker_Service SHALL NOT modify artifact data (pass through from ML_Service)

### Requirement 9: Service Separation & Deployment

**User Story:** As a system operator, I want API and Worker services to run independently, so that I can scale them separately and deploy updates without downtime.

#### Acceptance Criteria

1. THE API_Service SHALL run as separate Docker container from Worker_Service
2. THE ML_Service SHALL run as separate Docker container from API_Service and Worker_Service
3. WHEN the API_Service starts, THE API_Service SHALL NOT start arq consumer (no job processing)
4. WHEN the Worker_Service starts, THE Worker_Service SHALL NOT expose HTTP endpoints
5. THE API_Service Dockerfile SHALL NOT include ML dependencies (smaller image)
6. THE Worker_Service Dockerfile SHALL include arq and httpx but NOT ML model files
7. THE ML_Service Dockerfile SHALL include all ML dependencies and model files
8. WHEN services are deployed, THE services SHALL communicate via internal Docker network (eioku-network)
9. THE ML_Service SHALL NOT be exposed to external clients (internal only)

### Requirement 10: Job Cancellation & Manual Management

**User Story:** As a system operator, I want to manually cancel long-running tasks, so that I can manage stuck or unwanted jobs.

#### Acceptance Criteria

1. WHEN an operator calls POST /tasks/{task_id}/cancel, THE API_Service SHALL mark task as CANCELLED in PostgreSQL
2. WHEN an operator calls POST /tasks/{task_id}/cancel, THE API_Service SHALL call Job.abort() on the arq job
3. IF a job is queued, THEN Job.abort() SHALL prevent it from being picked up
4. IF a job is running, THEN Job.abort() SHALL raise asyncio.CancelledError inside the worker
5. WHEN a Worker_Service catches CancelledError, THE Worker_Service SHALL mark task as CANCELLED and re-raise the exception
6. WHEN an operator calls POST /tasks/{task_id}/retry, THE API_Service SHALL reset task to PENDING and re-enqueue
7. WHEN an operator calls GET /tasks, THE API_Service SHALL support filtering by status, task_type, video_id
8. WHEN an operator calls GET /tasks, THE API_Service SHALL support sorting by created_at, started_at, running_time

### Requirement 11: Observability & Logging

**User Story:** As a system operator, I want structured logs and metrics, so that I can monitor system health and debug issues.

#### Acceptance Criteria

1. WHEN services start, THE services SHALL emit structured logs with timestamps and log levels
2. WHEN a job is enqueued, THE API_Service SHALL log job_id and task_id
3. WHEN a job is consumed, THE Worker_Service SHALL log job_id and task_id
4. WHEN a job completes, THE Worker_Service SHALL log job_id, task_id, and artifact_count
5. WHEN a job fails, THE Worker_Service SHALL log job_id, task_id, and error message
6. WHEN the Reconciler runs, THE Reconciler SHALL log reconciliation actions (re-enqueued, synced, alerted)
7. THE system SHALL track Redis queue depth metrics
8. THE system SHALL track job processing latency histogram
9. THE system SHALL track ML_Service GPU utilization
10. THE system SHALL track failed job rate

### Requirement 12: Configuration & Environment

**User Story:** As a system operator, I want to configure services via environment variables, so that I can deploy to different environments without code changes.

#### Acceptance Criteria

1. THE API_Service SHALL read DATABASE_URL from environment for PostgreSQL connection
2. THE API_Service SHALL read REDIS_URL from environment for Redis connection
3. THE API_Service SHALL read ML_SERVICE_URL from environment for ML Service base URL
4. THE Worker_Service SHALL read DATABASE_URL from environment for PostgreSQL connection
5. THE Worker_Service SHALL read REDIS_URL from environment for Redis connection
6. THE Worker_Service SHALL read ML_SERVICE_URL from environment for ML Service base URL
7. THE Worker_Service SHALL read ARQ_MAX_JOBS from environment (default=4)
8. THE Worker_Service SHALL read ARQ_JOB_TIMEOUT from environment (default=1800 seconds)
9. THE ML_Service SHALL read GPU_CONCURRENCY from environment (default=2)
10. THE ML_Service SHALL read MODEL_CACHE_DIR from environment for model storage

### Requirement 13: Backward Compatibility

**User Story:** As a system operator, I want the migration to be non-breaking, so that existing clients and workflows continue to work.

#### Acceptance Criteria

1. WHEN the system is deployed, THE existing REST API endpoints SHALL remain unchanged
2. WHEN a client queries task status, THE response format SHALL remain compatible
3. WHEN a client queries artifacts, THE response format SHALL remain compatible
4. WHEN the system is deployed, THE existing database schema SHALL NOT require changes
5. WHEN the system is deployed, THE existing video discovery workflow SHALL continue to work
6. WHEN the system is deployed, THE existing artifact navigation and search SHALL continue to work

### Requirement 14: Error Handling & Resilience

**User Story:** As a system operator, I want the system to handle errors gracefully, so that failures don't cascade or lose data.

#### Acceptance Criteria

1. WHEN the ML_Service is unavailable, THE Worker_Service SHALL retry the job with exponential backoff
2. WHEN the ML_Service returns an error, THE Worker_Service SHALL mark task as FAILED with error message
3. WHEN PostgreSQL is unavailable, THE Worker_Service SHALL retry the job (arq handles this)
4. WHEN Redis is unavailable, THE API_Service SHALL fail gracefully with appropriate error response
5. WHEN a Worker_Service crashes mid-job, THE Reconciler SHALL detect and re-enqueue the job
6. WHEN an artifact batch insert fails, THE Worker_Service SHALL mark task as FAILED and not acknowledge the job
7. WHEN a job exceeds max_tries, THE Worker_Service SHALL mark task as FAILED and stop retrying

### Requirement 15: Performance & Scalability

**User Story:** As a system operator, I want the system to scale horizontally, so that I can handle increased load.

#### Acceptance Criteria

1. WHEN multiple Worker_Service instances are deployed, THE instances SHALL consume jobs from the same Redis queue
2. WHEN multiple Worker_Service instances are deployed, THE instances SHALL NOT process the same job twice (arq consumer groups)
3. WHEN multiple ML_Service instances are deployed, THE instances SHALL share GPU resources via semaphore (or separate GPUs)
4. WHEN the API_Service receives requests, THE API_Service SHALL handle concurrent requests via connection pooling
5. THE job queue overhead SHALL be negligible (<100ms per job)
6. THE database operations SHALL NOT block job processing

### Requirement 16: Data Integrity & Consistency

**User Story:** As a system operator, I want data to remain consistent across services, so that the system is reliable and auditable.

#### Acceptance Criteria

1. WHEN artifacts are inserted, THE artifacts table transaction SHALL be atomic (all-or-nothing)
2. WHEN a task status is updated, THE update SHALL be immediately visible to other services
3. WHEN a job is acknowledged in Redis, THE corresponding task SHALL be marked COMPLETED in PostgreSQL
4. WHEN a job fails, THE task error message SHALL be persisted in PostgreSQL
5. WHEN the Reconciler syncs state, THE PostgreSQL state SHALL be the source of truth
6. WHEN artifacts are inserted, THE projection tables SHALL be updated via database triggers (existing behavior)

### Requirement 17: ML Service Model Management

**User Story:** As a system operator, I want ML models to be downloaded and verified on startup, so that the service is ready to serve requests immediately and I can detect GPU issues early.

#### Acceptance Criteria

1. WHEN the ML_Service starts, THE ML_Service SHALL download all required models (object detection, face detection, transcription, OCR, place detection, scene detection)
2. WHEN the ML_Service downloads models, THE ML_Service SHALL cache them locally to avoid re-downloading
3. WHEN the ML_Service starts, THE ML_Service SHALL verify that each model loads successfully
4. WHEN the ML_Service starts, THE ML_Service SHALL detect GPU availability via torch.cuda.is_available()
5. WHEN the ML_Service detects GPU availability, THE ML_Service SHALL log GPU device name, device count, and total memory
6. WHEN the ML_Service starts, THE ML_Service SHALL verify that each model can detect GPU or CPU correctly
7. IF GPU is not available, THE ML_Service SHALL log a warning and continue with CPU-only mode (unless REQUIRE_GPU=true)
8. IF REQUIRE_GPU environment variable is set to true AND GPU is not available, THEN THE ML_Service SHALL fail startup
9. WHEN the ML_Service receives GET /health request, THE ML_Service SHALL return model status (ready or failed) for each model
10. WHEN the ML_Service receives GET /health request, THE ML_Service SHALL return GPU availability, device info, and memory usage
11. IF any model failed to initialize, THE GET /health endpoint SHALL return status=degraded or unhealthy

### Requirement 18: HTTP Client & Communication

**User Story:** As a Worker Service, I want reliable HTTP communication with the ML Service, so that inference requests are robust and efficient.

#### Acceptance Criteria

1. THE Worker_Service SHALL use httpx async client for HTTP requests to ML_Service
2. WHEN the Worker_Service makes HTTP requests, THE Worker_Service SHALL use connection pooling
3. WHEN the Worker_Service makes HTTP requests, THE Worker_Service SHALL set timeout to 600 seconds (for long-running inference)
4. WHEN the ML_Service is slow to respond, THE Worker_Service SHALL wait without blocking other jobs
5. WHEN an HTTP request fails, THE Worker_Service SHALL allow arq to retry the job

### Requirement 19: Job Deduplication & Idempotency

**User Story:** As a system operator, I want jobs to be deduplicated, so that duplicate task executions don't occur.

#### Acceptance Criteria

1. WHEN a job is enqueued, THE job_id SHALL be set to "ml_{task_id}" for deduplication
2. WHEN a duplicate job is enqueued, THE arq queue SHALL recognize it as duplicate and not create a new job
3. WHEN a Worker_Service processes a job, THE Worker_Service SHALL check if task is already COMPLETED before processing
4. IF a task is already COMPLETED, THEN THE Worker_Service SHALL skip processing and return success

### Requirement 20: Graceful Shutdown

**User Story:** As a system operator, I want services to shut down gracefully, so that in-flight jobs are handled properly.

#### Acceptance Criteria

1. WHEN the Worker_Service receives shutdown signal, THE Worker_Service SHALL finish current job before exiting
2. WHEN the Worker_Service receives shutdown signal, THE Worker_Service SHALL NOT acknowledge incomplete jobs
3. WHEN the Worker_Service shuts down, THE incomplete jobs SHALL remain in Redis for another worker to pick up
4. WHEN the API_Service receives shutdown signal, THE API_Service SHALL close database connections gracefully
5. WHEN the ML_Service receives shutdown signal, THE ML_Service SHALL unload models and close connections

### Requirement 21: ML Service Response Models

**User Story:** As a Worker Service, I want to receive structured responses from the ML Service, so that I can reliably transform and persist results.

#### Acceptance Criteria

1. THE ML_Service SHALL define Pydantic response models for each inference endpoint
2. WHEN the ML_Service returns results, THE response models SHALL include run_id, config_hash, input_hash, model_profile, producer, and producer_version
3. WHEN the ML_Service returns object detection results, THE response model SHALL include list of Detection objects with frame_index, timestamp_ms, label, confidence, and bounding_box
4. WHEN the ML_Service returns face detection results, THE response model SHALL include list of Detection objects with cluster_id
5. WHEN the ML_Service returns transcription results, THE response model SHALL include list of Segment objects with start_ms, end_ms, text, confidence, and words
6. WHEN the ML_Service returns OCR results, THE response model SHALL include list of Detection objects with polygon coordinates
7. WHEN the ML_Service returns place classification results, THE response model SHALL include list of Classification objects with frame_index, timestamp_ms, and predictions
8. WHEN the ML_Service returns scene detection results, THE response model SHALL include list of Scene objects with scene_index, start_ms, and end_ms
9. THE ML_Service response models SHALL be defined in ml-service/src/models/responses.py
10. THE ML_Service response models SHALL be validated by Pydantic on output

### Requirement 22: Artifact Envelope Transformation

**User Story:** As a Worker Service, I want to transform ML responses into ArtifactEnvelopes, so that results are persisted in the correct format.

#### Acceptance Criteria

1. WHEN the Worker_Service receives ML response, THE Worker_Service SHALL extract individual detections/segments from the batch response
2. FOR each detection/segment, THE Worker_Service SHALL create an ArtifactEnvelope with artifact_type, span_start_ms, span_end_ms, and payload_json
3. WHEN creating ArtifactEnvelope, THE Worker_Service SHALL copy config_hash, input_hash, producer, producer_version, model_profile, and run_id from ML response
4. WHEN creating ArtifactEnvelope, THE Worker_Service SHALL set payload_json to contain the individual detection/segment data
5. WHEN the Worker_Service persists artifacts, THE Worker_Service SHALL batch insert all ArtifactEnvelopes for a task in a single transaction
6. THE ArtifactEnvelope transformation logic SHALL be defined in Worker_Service

### Requirement 23: Artifact Payload Schema Validation

**User Story:** As an API Service, I want to validate artifact payloads, so that data integrity is maintained in the database.

#### Acceptance Criteria

1. THE API_Service SHALL define Pydantic schema models for each artifact type (ObjectDetectionV1, TranscriptSegmentV1, etc.)
2. WHEN artifacts are inserted into PostgreSQL, THE API_Service SHALL validate payload_json against the corresponding schema model
3. WHEN payload_json validation fails, THE API_Service SHALL reject the insert and return error
4. THE artifact schema models SHALL be defined in backend/src/domain/schemas/
5. WHEN an artifact is queried, THE API_Service MAY deserialize payload_json using the schema model for type safety
6. THE schema models SHALL be versioned (e.g., ObjectDetectionV1, ObjectDetectionV2) to support schema evolution

