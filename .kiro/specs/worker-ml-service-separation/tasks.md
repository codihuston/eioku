# Implementation Plan: Worker & ML Service Separation

## Overview

This implementation plan breaks down the service separation into discrete, testable tasks. The approach follows incremental development with frequent validation at each step. Tasks are organized into phases that can be executed sequentially, with each phase building on the previous one.

**Current Status**: Task orchestration and handlers are implemented in-process. This plan focuses on extracting ML models to a separate service and implementing arq-based job queue infrastructure.

## Phase 1: Infrastructure Setup (Days 1-2)

### 1. Add dependencies and create Redis service

- [x] 1.1 Add arq, httpx to backend/pyproject.toml
  - Add arq ^0.26 for job queue
  - Add httpx ^0.25 for async HTTP client
  - _Requirements: 4.1, 18.1_

- [x] 1.2 Update docker-compose.yml to add Valkey service
  - Add valkey service (port 6379)
  - Configure AOF persistence
  - Add eioku-network for service communication
  - _Requirements: 4.10, 4.11_

- [ ]* 1.3 Write unit test for Redis connection
  - Test Valkey service is accessible
  - Test connection pooling works
  - _Requirements: 4.1_

### 2. Create ML Service project structure

- [x] 2.1 Create ml-service directory with Python project layout
  - Create ml-service/src/main.py (FastAPI app)
  - Create ml-service/src/api/ (endpoint routers)
  - Create ml-service/src/services/ (ML service implementations)
  - Create ml-service/src/models/ (Pydantic request/response models)
  - Create ml-service/src/utils/ (GPU utilities, provenance hashing)
  - Create ml-service/pyproject.toml with dependencies
  - Create ml-service/Dockerfile
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ]* 2.2 Write unit tests for ML Service project structure
  - Test that FastAPI app initializes
  - Test that routers are registered
  - _Requirements: 3.1_

### 3. Implement model initialization and GPU detection

- [x] 3.1 Create ModelManager for downloading and verifying models
  - Implement download_model() for each model type (YOLO, Whisper, Places365, EasyOCR)
  - Implement verify_model() to test model loading
  - Implement GPU detection via torch.cuda.is_available()
  - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_

- [x] 3.2 Implement FastAPI lifespan with model initialization
  - Download all models on startup
  - Verify GPU availability and log device info
  - Handle graceful degradation (CPU fallback unless REQUIRE_GPU=true)
  - Fail startup if any model fails to initialize
  - _Requirements: 17.1, 17.7, 17.8_

- [ ]* 3.3 Write property test for model initialization
  - **Property 13: Model Initialization on Boot**
  - **Validates: Requirements 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7**

### 4. Implement inference endpoints

- [x] 4.1 Implement object detection endpoint (/infer/objects)
  - Create ObjectDetectionRequest and ObjectDetectionResponse models
  - Implement GPU semaphore acquisition
  - Call ObjectDetectionService and return structured response
  - Include config_hash, input_hash, run_id in response
  - _Requirements: 3.1, 3.8, 3.9, 5.1_

- [x] 4.2 Implement face detection endpoint (/infer/faces)
  - Create FaceDetectionRequest and FaceDetectionResponse models
  - Implement GPU semaphore acquisition
  - Call FaceDetectionService and return structured response
  - _Requirements: 3.2, 3.8, 3.9_

- [x] 4.3 Implement transcription endpoint (/infer/transcribe)
  - Create TranscriptionRequest and TranscriptionResponse models
  - Implement GPU semaphore acquisition
  - Call TranscriptionService and return structured response
  - _Requirements: 3.3, 3.8, 3.9_

- [x] 4.4 Implement OCR endpoint (/infer/ocr)
  - Create OCRRequest and OCRResponse models
  - Implement GPU semaphore acquisition
  - Call OCRService and return structured response
  - _Requirements: 3.4, 3.8, 3.9_

- [x] 4.5 Implement place detection endpoint (/infer/places)
  - Create PlaceDetectionRequest and PlaceDetectionResponse models
  - Implement GPU semaphore acquisition
  - Call PlaceDetectionService and return structured response
  - _Requirements: 3.5, 3.8, 3.9_

- [x] 4.6 Implement scene detection endpoint (/infer/scenes)
  - Create SceneDetectionRequest and SceneDetectionResponse models
  - Implement GPU semaphore acquisition
  - Call SceneDetectionService and return structured response
  - _Requirements: 3.6, 3.8, 3.9_

- [ ]* 4.7 Write property test for inference endpoints
  - **Property 5: ML Service Response Provenance**
  - **Validates: Requirements 3.8, 3.9, 8.1**

### 5. Implement health check endpoint

- [ ] 5.1 Implement GET /health endpoint
  - Return model status (ready/failed) for each model
  - Return GPU availability and device info
  - Return memory usage (allocated, reserved, total)
  - Return degraded status if any model failed
  - _Requirements: 3.7, 17.8, 17.9, 17.10_

- [ ]* 5.2 Write unit test for health endpoint
  - Test healthy response when all models ready
  - Test degraded response when model fails
  - Test GPU info is included
  - _Requirements: 3.7_

### 6. Checkpoint - ML Service Complete

- [ ] 6.1 Ensure all ML Service tests pass
  - Run pytest on ml-service/tests/
  - Verify all endpoints respond correctly
  - Verify GPU semaphore limits concurrent requests
  - Verify health endpoint returns correct status

---

## Phase 2: Job Queue Infrastructure (Days 2-3)

### 7. Create JobProducer with GPU/CPU routing

- [ ] 7.1 Create JobProducer class in backend/src/services/
  - Implement _get_queue_name() to route tasks to gpu_jobs or cpu_jobs
  - Implement enqueue_task() to enqueue to appropriate queue
  - Map task types to GPU/CPU requirements
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ]* 7.2 Write unit test for JobProducer
  - Test GPU task routing to gpu_jobs
  - Test CPU task routing to cpu_jobs
  - Test job_id format "ml_{task_id}"
  - _Requirements: 4.1, 4.5_

### 8. Create ML Client for Worker Service

- [ ] 8.1 Create MLClient class in backend/src/services/
  - Implement async httpx client with connection pooling
  - Implement infer() method for each task type
  - Set timeout to 600 seconds for long-running inference
  - Handle HTTP errors and retry logic
  - _Requirements: 18.1, 18.2, 18.3, 18.4_

- [ ]* 8.2 Write unit test for MLClient
  - Test HTTP requests to ML Service
  - Test response parsing
  - Test timeout handling
  - _Requirements: 18.1_

### 9. Create arq worker configuration

- [ ] 9.1 Create arq worker configuration in backend/src/workers/
  - Read GPU_MODE environment variable (gpu, cpu, auto)
  - Implement get_queue_names() to select appropriate queue
  - Auto-detect GPU if GPU_MODE=auto
  - Configure max_jobs, job_timeout, max_tries
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ]* 9.2 Write unit test for worker configuration
  - Test GPU mode selects gpu_jobs queue
  - Test CPU mode selects cpu_jobs queue
  - Test auto mode detects GPU correctly
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

### 10. Implement task handler for arq

- [ ] 10.1 Create process_ml_task() handler in backend/src/workers/
  - Pre-flight check: verify task status is not COMPLETED/CANCELLED
  - Update task status to RUNNING
  - Call MLClient.infer() to dispatch to ML Service
  - Handle asyncio.CancelledError for task cancellation
  - Handle other exceptions and mark task as FAILED
  - _Requirements: 2.6, 2.7, 5.2, 5.3, 14.1, 20.1_

- [ ] 10.2 Create artifact transformation logic
  - Transform ML response to ArtifactEnvelopes
  - Extract individual detections/segments from batch response
  - Copy provenance metadata (config_hash, input_hash, etc.)
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 22.1, 22.2, 22.3, 22.4, 22.5_

- [ ] 10.3 Create artifact persistence logic
  - Batch insert ArtifactEnvelopes to PostgreSQL
  - Update task status to COMPLETED
  - Acknowledge job in Redis (XACK)
  - _Requirements: 2.8, 2.9, 2.10, 8.2, 16.1_

- [ ]* 10.4 Write property test for task handler
  - **Property 4: Worker Job Consumption and Acknowledgment**
  - **Validates: Requirements 2.1, 2.6, 5.2, 5.3**

### 11. Implement reconciler

- [ ] 11.1 Create Reconciler class in backend/src/workers/
  - Implement _sync_pending_tasks() to re-enqueue orphaned PENDING tasks
  - Implement _sync_running_tasks() to sync RUNNING tasks with Redis state
  - Implement _alert_long_running_tasks() to alert (never auto-kill)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10_

- [ ] 11.2 Add reconciler as periodic cron task in arq
  - Schedule reconciler to run every 5 minutes
  - _Requirements: 6.7_

- [ ]* 11.3 Write property test for reconciliation
  - **Property 7: Reconciliation Recovery**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6**

### 12. Checkpoint - Job Queue Complete

- [ ] 12.1 Ensure all job queue tests pass
  - Run pytest on backend/tests/workers/
  - Verify job consumption from Redis
  - Verify artifact persistence
  - Verify reconciliation recovery

---

## Phase 3: API Service Updates (Days 3-4)

### 13. Update video discovery to auto-create tasks

- [ ] 13.1 Create VideoDiscoveryService
  - Implement discover_and_queue_tasks() to auto-create 6 tasks per video
  - Implement _get_default_config() for each task type
  - Call JobProducer to enqueue tasks to appropriate queues
  - _Requirements: 1.2, 1.3_

- [ ] 13.2 Integrate VideoDiscoveryService into video discovery workflow
  - Update existing video discovery to call VideoDiscoveryService
  - Ensure tasks are auto-created and auto-queued on video discovery
  - _Requirements: 1.2, 1.3_

- [ ]* 13.3 Write property test for video discovery
  - **Property 1: Video Discovery Auto-Creates Tasks**
  - **Validates: Requirements 1.2, 1.3**

### 14. Add manual task enqueueing endpoint

- [ ] 14.1 Create POST /tasks/{task_id}/enqueue endpoint
  - Verify task exists and is in PENDING status
  - Call JobProducer to enqueue task
  - Return job_id and status
  - _Requirements: 1.3_

- [ ]* 14.2 Write unit test for manual enqueueing endpoint
  - Test successful enqueueing
  - Test error when task not found
  - Test error when task not in PENDING status
  - _Requirements: 1.3_

### 15. Add task cancellation endpoints

- [ ] 15.1 Create POST /tasks/{task_id}/cancel endpoint
  - Mark task as CANCELLED in PostgreSQL
  - Call Job.abort() on arq job
  - Return cancellation status
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 15.2 Create POST /tasks/{task_id}/retry endpoint
  - Verify task is in FAILED or CANCELLED status
  - Reset task to PENDING
  - Re-enqueue to appropriate queue
  - _Requirements: 10.1_

- [ ] 15.3 Create GET /tasks endpoint with filtering and sorting
  - Support filtering by status, task_type, video_id
  - Support sorting by created_at, started_at, running_time
  - Return task list with pagination
  - _Requirements: 10.7, 10.8_

- [ ]* 15.4 Write unit tests for task management endpoints
  - Test cancellation
  - Test retry
  - Test filtering and sorting
  - _Requirements: 10.1, 10.7, 10.8_

### 16. Add artifact schema validation

- [ ] 16.1 Create artifact payload schema models
  - Create ObjectDetectionV1 schema
  - Create FaceDetectionV1 schema
  - Create TranscriptSegmentV1 schema
  - Create OCRDetectionV1 schema
  - Create PlaceClassificationV1 schema
  - Create SceneV1 schema
  - _Requirements: 23.1, 23.2, 23.6_

- [ ] 16.2 Integrate schema validation into artifact persistence
  - Validate payload_json against schema model on insert
  - Reject invalid payloads with error message
  - _Requirements: 23.3, 23.4, 23.5_

- [ ]* 16.3 Write unit tests for artifact schema validation
  - Test valid payloads pass validation
  - Test invalid payloads are rejected
  - _Requirements: 23.3, 23.4_

### 17. Checkpoint - API Service Complete

- [ ] 17.1 Ensure all API Service tests pass
  - Run pytest on backend/tests/api/
  - Verify video discovery auto-creates tasks
  - Verify manual enqueueing works
  - Verify task cancellation works
  - Verify artifact validation works

---

## Phase 4: Service Separation & Deployment (Days 4-5)

### 18. Create separate entry points

- [ ] 18.1 Create backend/src/main_api.py
  - FastAPI app for API Service only
  - No arq consumer initialization
  - Database connection pool (max_size=20)
  - Redis connection for job enqueueing
  - _Requirements: 9.3, 9.4_

- [ ] 18.2 Create backend/src/main_worker.py
  - arq worker entry point
  - No HTTP endpoints
  - Database connection pool (max_size=10)
  - Redis connection for job consumption
  - _Requirements: 9.3, 9.4_

- [ ]* 18.3 Write unit tests for entry points
  - Test API Service doesn't start arq consumer
  - Test Worker Service doesn't expose HTTP endpoints
  - _Requirements: 9.3, 9.4_

### 19. Create separate Dockerfiles

- [ ] 19.1 Create dev/Dockerfile.api
  - Lighter image without ML dependencies
  - Install only FastAPI, uvicorn, database drivers
  - Expose port 8000
  - _Requirements: 9.5_

- [ ] 19.2 Create dev/Dockerfile.worker
  - Include arq, httpx, database drivers
  - No ML model files
  - No HTTP server
  - _Requirements: 9.6_

- [ ] 19.3 Create dev/Dockerfile.ml
  - Include all ML dependencies
  - Include model cache directory
  - _Requirements: 9.7_

### 20. Update docker-compose.yml

- [ ] 20.1 Update docker-compose.yml with separate services
  - Add api service (port 8000)
  - Add worker-gpu service (GPU_MODE=gpu)
  - Add worker-cpu service (GPU_MODE=cpu)
  - Add ml-service (port 8001)
  - Add valkey service (Redis replacement)
  - Update nginx to route to api service
  - _Requirements: 9.1, 9.2, 9.8, 9.9_

- [ ]* 20.2 Write integration test for docker-compose
  - Test all services start correctly
  - Test services can communicate
  - Test health endpoints respond
  - _Requirements: 9.1, 9.2_

### 21. Implement backward compatibility

- [ ] 21.1 Verify all existing REST endpoints still work
  - Test GET /videos
  - Test GET /tasks
  - Test GET /artifacts
  - Test GET /paths
  - Test POST endpoints
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

- [ ]* 21.2 Write property test for backward compatibility
  - **Property 12: Backward Compatibility**
  - **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.6**

### 22. Implement observability

- [ ] 22.1 Add structured logging to all services
  - Log service startup with configuration
  - Log job enqueueing with task_id and queue
  - Log job consumption with task_id
  - Log job completion with artifact_count
  - Log job failures with error message
  - Log reconciliation actions
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ]* 22.2 Write unit tests for logging
  - Test logs contain required fields
  - Test logs are structured (JSON format)
  - _Requirements: 11.1_

### 23. Checkpoint - Service Separation Complete

- [ ] 23.1 Ensure all integration tests pass
  - Run full test suite
  - Verify all services work together
  - Verify backward compatibility
  - Verify error handling

---

## Phase 5: Advanced Features (Days 5-6)

### 24. Implement GPU semaphore concurrency control

- [ ] 24.1 Verify GPU semaphore limits concurrent requests
  - Test that max GPU_CONCURRENCY requests run simultaneously
  - Test that excess requests are queued
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ]* 24.2 Write property test for GPU semaphore
  - **Property 8: GPU Semaphore Concurrency Control**
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

### 25. Implement horizontal scaling

- [ ] 25.1 Test multiple Worker Service instances
  - Deploy multiple worker-gpu instances
  - Deploy multiple worker-cpu instances
  - Verify jobs are distributed without duplicates
  - _Requirements: 15.1, 15.2_

- [ ]* 25.2 Write property test for horizontal scaling
  - **Property 11: Horizontal Scaling**
  - **Validates: Requirements 15.1, 15.2**

### 26. Implement error handling and resilience

- [ ] 26.1 Test error scenarios
  - Test ML Service unavailable → retry with backoff
  - Test PostgreSQL unavailable → retry
  - Test Redis unavailable → fail gracefully
  - Test artifact insert failure → mark task as FAILED
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_

- [ ]* 26.2 Write property test for error handling
  - **Property 10: Error Handling and Retry**
  - **Validates: Requirements 14.1, 14.2, 14.3, 14.7**

### 27. Implement graceful shutdown

- [ ] 27.1 Test graceful shutdown behavior
  - Send SIGTERM to Worker Service
  - Verify current job completes
  - Verify incomplete jobs remain in Redis
  - Verify another worker picks up incomplete job
  - _Requirements: 20.1, 20.2, 20.3_

- [ ]* 27.2 Write property test for graceful shutdown
  - **Property 15: Graceful Shutdown**
  - **Validates: Requirements 20.1, 20.2, 20.3**

### 28. Implement job idempotency

- [ ] 28.1 Test job idempotency
  - Enqueue same job twice
  - Verify only one job is processed
  - Verify duplicate job is deduplicated
  - _Requirements: 19.1, 19.2, 19.3_

- [ ]* 28.2 Write property test for job idempotency
  - **Property 14: Job Idempotency**
  - **Validates: Requirements 19.2, 19.3**

### 29. Implement service isolation

- [ ] 29.1 Verify service isolation
  - Verify API Service doesn't run arq consumer
  - Verify Worker Service doesn't expose HTTP endpoints
  - Verify ML Service doesn't access PostgreSQL
  - _Requirements: 9.3, 9.4, 9.10, 9.11_

- [ ]* 29.2 Write property test for service isolation
  - **Property 16: Service Isolation**
  - **Validates: Requirements 9.3, 9.4, 9.10, 9.11**

### 30. Final checkpoint - All features complete

- [ ] 30.1 Run full test suite
  - Run all unit tests
  - Run all property-based tests
  - Run all integration tests
  - Verify all tests pass

- [ ] 30.2 Verify all requirements are met
  - Check each requirement has corresponding test
  - Check each property is implemented
  - Check backward compatibility maintained

- [ ] 30.3 Documentation and cleanup
  - Update README with new architecture
  - Document environment variables
  - Document deployment procedures
  - Clean up any temporary code

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task should be independently testable
- Property-based tests should run minimum 100 iterations
- All tests must pass before proceeding to next phase
- Checkpoints ensure incremental validation
- Backward compatibility must be maintained throughout
