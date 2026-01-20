# Implementation Plan: Semantic Video Search Platform (Eioku)

## Overview

This implementation plan breaks down the Eioku semantic video search platform into discrete, actionable tasks. The plan follows an incremental approach, building core functionality first (video processing and search), then adding the user interface, and finally implementing advanced features.

## Tasks

- [ ] 1. Set up project structure and development environment
  - Create Python backend project with poetry or pip-tools
  - Create Electron + React + TypeScript frontend project
  - Configure development tools (linting, formatting, type checking)
  - Set up testing framework (pytest, jest)
  - Create Dockerfile for backend
  - Set up GitHub Actions for CI/CD (build, test, Docker image)
  - _Requirements: 1.1, 1.2, 1.3, 7.1, 7.2_

- [ ] 2. Create database schema and migrations
  - [ ] 2.1 Create Videos table schema
    - Define columns: videoId, filePath, filename, duration, fileSize, processedAt, lastModified, status
    - Add indexes on filePath and status
    - _Requirements: 1.1, 4.1, 4.4_
  
  - [ ] 2.2 Create Transcriptions table schema
    - Define columns: segmentId, videoId, text, start, end, confidence, speaker
    - Add foreign key to Videos table
    - Add index on videoId
    - _Requirements: 1.4, 1.5_
  
  - [ ] 2.3 Create Scenes table schema
    - Define columns: sceneId, videoId, scene, start, end, thumbnailPath
    - Add foreign key to Videos table
    - Add index on videoId
    - _Requirements: 10.4_
  
  - [ ] 2.4 Create Objects table schema
    - Define columns: objectId, videoId, label, timestamps (JSON), boundingBoxes (JSON)
    - Add foreign key to Videos table
    - Add index on videoId and label
    - _Requirements: 10.6, 10.7_
  
  - [ ] 2.5 Create Faces table schema
    - Define columns: faceId, videoId, personId, timestamps (JSON), boundingBoxes (JSON), confidence
    - Add foreign key to Videos table
    - Add index on videoId
    - _Requirements: 10.9, 10.10_
  
  - [ ] 2.6 Create Topics table schema
    - Define columns: topicId, videoId, label, keywords (JSON), relevanceScore, timestamps (JSON)
    - Add foreign key to Videos table
    - Add index on videoId and label
    - _Requirements: 9.1, 9.2_
  
  - [ ] 2.7 Create PathConfigs table schema
    - Define columns: pathId, path, recursive, addedAt
    - Add unique constraint on path
    - _Requirements: 1.1, 4.7_
  
  - [ ] 2.8 Create Tasks table schema
    - Define columns: taskId, videoId, taskType, status, priority, dependencies (JSON), createdAt, startedAt, completedAt, error
    - Add foreign key to Videos table
    - Add indexes on videoId, taskType, status
    - _Requirements: 6.2, 6.5_
  
  - [ ] 2.9 Create database migration scripts
    - Implement schema versioning
    - Create initial migration
    - _Requirements: 7.1, 7.2_

- [ ] 3. Implement database access layer
  - [ ] 3.1 Create Video DAO (Data Access Object)
    - Implement CRUD operations for Videos table
    - Implement query methods (by status, by path, etc.)
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [ ] 3.2 Create Transcription DAO
    - Implement CRUD operations for Transcriptions table
    - Implement query methods (by videoId, by time range)
    - _Requirements: 1.4, 1.5_
  
  - [ ] 3.3 Create Scene DAO
    - Implement CRUD operations for Scenes table
    - Implement query methods (by videoId)
    - _Requirements: 10.4_
  
  - [ ] 3.4 Create Object DAO
    - Implement CRUD operations for Objects table
    - Implement query methods (by videoId, by label)
    - _Requirements: 10.6, 10.7_
  
  - [ ] 3.5 Create Face DAO
    - Implement CRUD operations for Faces table
    - Implement query methods (by videoId, by personId)
    - _Requirements: 10.9, 10.10_
  
  - [ ] 3.6 Create Topic DAO
    - Implement CRUD operations for Topics table
    - Implement query methods (by videoId, aggregate across videos)
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ] 3.7 Create PathConfig DAO
    - Implement CRUD operations for PathConfigs table
    - Implement query methods (list all paths)
    - _Requirements: 1.1, 4.7_
  
  - [ ] 3.8 Create Task DAO
    - Implement CRUD operations for Tasks table
    - Implement query methods (by videoId, by status, by taskType)
    - _Requirements: 6.2, 6.5_
  
  - [ ] 3.9 Implement database connection management
    - Create connection pool
    - Implement transaction management
    - Handle connection errors
    - _Requirements: 7.1, 7.2, 7.4_
  
  - [ ]* 3.10 Write unit tests for all DAOs
    - Test CRUD operations for each DAO
    - Test transaction rollback
    - Test constraint violations
    - _Requirements: 4.1, 4.2, 4.3_

- [ ] 4. Implement path management and video discovery
  - [ ] 4.1 Create path configuration manager
    - Implement add/remove/list path operations
    - Store path configurations in database
    - _Requirements: 1.1, 1.2, 4.7_
  
  - [ ] 4.2 Implement video file discovery
    - Scan folders recursively for video files
    - Filter by supported formats (MP4, MOV, AVI, MKV)
    - Detect duplicate videos by file path
    - _Requirements: 1.1, 1.3, 1.8, 4.3_
  
  - [ ] 4.3 Implement file validation and missing file detection
    - Check if video files exist at stored paths
    - Flag missing or moved files
    - _Requirements: 4.5_
  
  - [ ]* 4.4 Write property test for path discovery
    - **Property 1: Path discovery completeness**
    - **Validates: Requirements 1.1, 1.8**
  
  - [ ]* 4.5 Write property test for individual file processing
    - **Property 2: Individual file processing**
    - **Validates: Requirements 1.2**

- [ ] 5. Implement task orchestration system
  - [ ] 5.1 Create task data models and queue structures
    - Define Task model with status, priority, dependencies
    - Implement in-memory task queues per task type
    - _Requirements: 6.2, 6.5_
  
  - [ ] 5.2 Implement task orchestrator
    - Create tasks for each video (transcription, scene, object, face)
    - Manage task dependencies (topic/embedding depend on transcription)
    - Track task completion and trigger dependent tasks
    - _Requirements: 1.4, 1.5, 1.6, 1.7, 6.2_
  
  - [ ] 5.3 Implement worker pool management
    - Create process pools for each task type
    - Implement worker assignment and task execution
    - Handle worker failures and task retries
    - _Requirements: 6.2, 6.5_
  
  - [ ] 5.4 Implement processing profile configuration
    - Load profile configurations (Balanced, Search First, Visual First, Low Resource)
    - Apply worker counts and priorities from profile
    - Allow custom profile creation
    - _Requirements: 6.2_
  
  - [ ]* 5.5 Write unit tests for task orchestration
    - Test task creation and dependency management
    - Test worker pool assignment
    - Test profile loading
    - _Requirements: 6.2, 6.5_

- [ ] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement video transcription pipeline
  - [ ] 7.1 Implement audio extraction from video
    - Use FFmpeg to extract audio track
    - Save to temporary file
    - _Requirements: 1.4_
  
  - [ ] 7.2 Implement Whisper transcription
    - Load Whisper model (large-v3 or turbo)
    - Transcribe audio with timestamps
    - Handle GPU acceleration if available
    - Generate speaker IDs for multi-speaker audio
    - _Requirements: 1.4, 1.5, 5.1, 5.3, 6.6_
  
  - [ ] 7.3 Store transcription segments in database
    - Save segments with start/end times
    - Store confidence scores
    - Flag low-confidence segments
    - _Requirements: 1.5, 5.5_
  
  - [ ]* 7.4 Write property test for transcription generation
    - **Property 4: Transcription generation with timestamps**
    - **Validates: Requirements 1.4, 1.5**
  
  - [ ]* 7.5 Write property test for confidence flagging
    - **Property 17: Confidence flagging**
    - **Validates: Requirements 5.5**

- [ ] 8. Implement scene detection pipeline
  - [ ] 8.1 Implement PySceneDetect integration
    - Use ContentDetector to find scene boundaries
    - Convert timecodes to seconds
    - _Requirements: 10.4_
  
  - [ ] 8.2 Generate scene thumbnails
    - Extract frame at scene start using FFmpeg
    - Save thumbnail images
    - Store thumbnail paths in database
    - _Requirements: 3.1, 3.2_
  
  - [ ] 8.3 Store scene boundaries in database
    - Save scene start/end times
    - Link to video
    - _Requirements: 10.4_
  
  - [ ]* 8.4 Write property test for scene detection
    - **Property 23: Scene detection and storage**
    - **Validates: Requirements 10.4**

- [ ] 9. Implement object detection pipeline
  - [ ] 9.1 Implement YOLOv8 object detection
    - Load YOLOv8 model (yolov8n or yolov8s)
    - Sample frames at configured interval
    - Run detection on sampled frames
    - Handle GPU acceleration if available
    - _Requirements: 10.6, 6.6_
  
  - [ ] 9.2 Aggregate and store object detections
    - Group detections by object label
    - Store timestamps and bounding boxes
    - Link to video
    - _Requirements: 10.6, 10.7_
  
  - [ ]* 9.3 Write unit tests for object detection
    - Test frame sampling
    - Test detection aggregation
    - Test GPU fallback to CPU
    - _Requirements: 10.6, 10.7_

- [ ] 10. Implement face detection pipeline
  - [ ] 10.1 Implement YOLOv8 face detection
    - Load YOLOv8 face model (yolov8n-face.pt)
    - Sample frames at configured interval (seconds)
    - Run detection on sampled frames
    - Handle GPU acceleration if available
    - _Requirements: 10.9, 6.6_
  
  - [ ] 10.2 Store face detections in database
    - Store timestamps and bounding boxes
    - Link to video
    - _Requirements: 10.9, 10.10_
  
  - [ ]* 10.3 Write unit tests for face detection
    - Test frame sampling by time interval
    - Test detection storage
    - Test GPU fallback to CPU
    - _Requirements: 10.9, 10.10_

- [ ] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement topic extraction
  - [ ] 12.1 Implement topic extraction from transcriptions
    - Use KeyBERT or BERTopic for topic extraction
    - Extract keywords and themes
    - Rank topics by relevance and frequency
    - _Requirements: 9.1, 9.5_
  
  - [ ] 12.2 Store topics in database
    - Save topic labels and keywords
    - Store relevance scores
    - Link to video and timestamps
    - _Requirements: 9.1, 9.2_
  
  - [ ] 12.3 Implement topic aggregation across videos
    - Aggregate topics across all videos
    - Rank by frequency
    - _Requirements: 9.3_
  
  - [ ]* 12.4 Write property test for topic extraction
    - **Property 18: Topic extraction**
    - **Validates: Requirements 9.1**
  
  - [ ]* 12.5 Write property test for topic aggregation
    - **Property 19: Topic aggregation**
    - **Validates: Requirements 9.3**

- [ ] 13. Implement semantic search with FAISS
  - [ ] 13.1 Implement embedding generation
    - Load sentence-transformers model (all-MiniLM-L6-v2)
    - Generate embeddings for transcript segments
    - Batch process for efficiency
    - _Requirements: 2.1, 2.2_
  
  - [ ] 13.2 Implement FAISS vector store
    - Create FAISS index for embeddings
    - Add embeddings with metadata
    - Save/load index from file
    - Implement search with cosine similarity
    - _Requirements: 2.1, 2.2, 2.3, 7.1, 7.2_
  
  - [ ] 13.3 Implement semantic search engine
    - Generate query embedding
    - Search FAISS index
    - Rank results by relevance
    - Filter by video, date range, etc.
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [ ] 13.4 Implement search result formatting
    - Include video filename, timestamp, matched text
    - Include thumbnail path
    - Include relevance score
    - _Requirements: 3.1, 3.2, 3.5_
  
  - [ ]* 13.5 Write property test for search result relevance
    - **Property 7: Search result relevance**
    - **Validates: Requirements 2.1**
  
  - [ ]* 13.6 Write property test for synonym understanding
    - **Property 8: Synonym understanding**
    - **Validates: Requirements 2.2**
  
  - [ ]* 13.7 Write property test for timestamp accuracy
    - **Property 9: Timestamp accuracy**
    - **Validates: Requirements 2.3**

- [ ] 14. Implement error handling and recovery
  - [ ] 14.1 Implement error handling for video processing
    - Catch and log processing errors
    - Mark videos with error status
    - Continue processing other videos
    - _Requirements: 1.7, 8.3_
  
  - [ ] 14.2 Implement data corruption detection and recovery
    - Validate database integrity on startup
    - Attempt recovery for corrupted data
    - Notify user of corruption
    - _Requirements: 7.4_
  
  - [ ] 14.3 Implement disk space monitoring
    - Check available disk space before processing
    - Warn user when space is low
    - _Requirements: 7.5_
  
  - [ ]* 14.4 Write property test for error isolation
    - **Property 5: Error isolation**
    - **Validates: Requirements 1.7**

- [ ] 15. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Implement backend API layer
  - [ ] 16.1 Create FastAPI application
    - Set up FastAPI app with CORS
    - Define API routes for library, search, player
    - Implement request/response models
    - _Requirements: 8.1, 8.2_
  
  - [ ] 16.2 Implement library management endpoints
    - GET /videos - list all videos
    - POST /videos/add-path - add folder or file path
    - DELETE /videos/{id} - remove video from library
    - GET /videos/{id} - get video details
    - GET /topics - get aggregated topics
    - _Requirements: 4.1, 4.2, 4.7, 9.3_
  
  - [ ] 16.3 Implement search endpoints
    - POST /search - semantic search
    - GET /search/suggestions - alternative search terms
    - _Requirements: 2.1, 2.4_
  
  - [ ] 16.4 Implement player endpoints
    - GET /videos/{id}/transcription - get transcript segments
    - GET /videos/{id}/scenes - get scene boundaries
    - GET /videos/{id}/objects - get detected objects
    - GET /videos/{id}/faces - get detected faces
    - _Requirements: 10.2, 10.5, 10.6, 10.9_
  
  - [ ] 16.5 Implement processing status endpoints
    - GET /processing/status - get overall processing status
    - GET /videos/{id}/status - get video processing status
    - _Requirements: 1.6, 6.5_
  
  - [ ]* 16.6 Write integration tests for API endpoints
    - Test all endpoints with valid/invalid inputs
    - Test error responses
    - _Requirements: 8.1, 8.2, 8.3_

- [ ] 17. Implement Electron frontend application
  - [ ] 17.1 Set up Electron + React + TypeScript project
    - Configure Electron main process
    - Set up React with TypeScript
    - Configure IPC communication between main and renderer
    - Set up routing (React Router)
    - _Requirements: 8.1_
  
  - [ ] 17.2 Implement library view
    - Display all videos in grid or list
    - Show video metadata (filename, duration, date)
    - Show processing status for each video
    - Implement sorting and filtering
    - Display extracted topics per video
    - _Requirements: 4.1, 8.4, 9.2_
  
  - [ ] 17.3 Implement path configuration dialog
    - Add folder path with recursive option
    - Add individual file path
    - Remove configured paths
    - Show video discovery preview
    - _Requirements: 1.1, 1.2, 4.7_
  
  - [ ] 17.4 Implement search interface
    - Search input with real-time suggestions
    - Topic suggestions (clickable)
    - Search filters (date, video, duration)
    - _Requirements: 2.1, 2.4, 9.4_
  
  - [ ] 17.5 Implement result gallery
    - Grid display of search results
    - Show thumbnail, filename, timestamp, relevance score
    - Hover to preview video clip
    - Click to open in player view
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 17.6 Implement processing status view
    - Show overall queue progress
    - Show per-video task progress (transcription, scenes, objects, faces)
    - Show errors with details
    - _Requirements: 1.6, 6.5, 8.2_
  
  - [ ]* 17.7 Write UI component tests
    - Test library view rendering
    - Test search interface
    - Test result gallery
    - _Requirements: 8.1, 8.2_

- [ ] 18. Implement player view
  - [ ] 18.1 Implement video player component
    - HTML5 video player with controls
    - Load video from file path
    - Seek to specific timestamp
    - _Requirements: 10.1_
  
  - [ ] 18.2 Implement transcript panel
    - Display transcript segments
    - Highlight current segment during playback
    - Auto-scroll to current segment
    - Click segment to jump to timestamp
    - _Requirements: 10.2, 10.3_
  
  - [ ] 18.3 Implement scene navigation
    - Display scene list
    - Next/previous scene buttons
    - Jump to scene boundary on click
    - _Requirements: 10.5_
  
  - [ ] 18.4 Implement object navigation
    - Display detected objects for current scene
    - Show object occurrences list
    - Next/previous buttons for each object
    - Jump to timestamp on click
    - _Requirements: 10.6, 10.7, 10.8_
  
  - [ ] 18.5 Implement face navigation
    - Display detected faces
    - Show face occurrences list
    - Next/previous buttons for each face
    - Jump to timestamp on click
    - _Requirements: 10.9, 10.10, 10.11_
  
  - [ ] 18.6 Implement timeline markers
    - Show scene boundaries on timeline
    - Show object/face occurrences on timeline
    - Click marker to jump to timestamp
    - _Requirements: 10.5, 10.8, 10.11_
  
  - [ ]* 18.7 Write property test for transcript synchronization
    - **Property 21: Transcript synchronization**
    - **Validates: Requirements 10.2**
  
  - [ ]* 18.8 Write property test for transcript navigation
    - **Property 22: Transcript navigation**
    - **Validates: Requirements 10.3**
  
  - [ ]* 18.9 Write property test for scene navigation
    - **Property 24: Scene navigation**
    - **Validates: Requirements 10.5**
  
  - [ ]* 18.10 Write property test for object navigation
    - **Property 25: Object navigation**
    - **Validates: Requirements 10.7, 10.8**
  
  - [ ]* 18.11 Write property test for face navigation
    - **Property 26: Face navigation**
    - **Validates: Requirements 10.10, 10.11**

- [ ] 19. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Implement data persistence and recovery
  - [ ] 20.1 Implement application state persistence
    - Save window size/position
    - Save selected processing profile
    - Save UI preferences
    - _Requirements: 7.1, 7.2_
  
  - [ ] 20.2 Implement graceful shutdown
    - Save all pending data on close
    - Cancel running tasks
    - Clean up temporary files
    - _Requirements: 7.1_
  
  - [ ] 20.3 Implement startup recovery
    - Load library and indices on startup
    - Resume interrupted processing
    - Validate file paths
    - _Requirements: 7.2, 4.5_
  
  - [ ]* 20.4 Write property test for data persistence round-trip
    - **Property 15: Data persistence round-trip**
    - **Validates: Requirements 7.1, 7.2**

- [ ] 21. Implement library management operations
  - [ ] 21.1 Implement video removal
    - Remove video metadata from database
    - Remove embeddings from FAISS index
    - Remove thumbnails from file system
    - Preserve original video file
    - _Requirements: 4.2_
  
  - [ ] 21.2 Implement duplicate detection
    - Check file path before processing
    - Prevent redundant processing
    - _Requirements: 4.3_
  
  - [ ] 21.3 Implement incremental path updates
    - Detect new videos in configured paths
    - Detect modified videos (by file modification time)
    - Re-process modified videos
    - _Requirements: 4.6_
  
  - [ ]* 21.4 Write property test for deletion safety
    - **Property 12: Deletion safety**
    - **Validates: Requirements 4.2**
  
  - [ ]* 21.5 Write property test for duplicate detection
    - **Property 6: Duplicate detection**
    - **Validates: Requirements 4.3**

- [ ] 22. Implement UI interactions and feedback
  - [ ] 22.1 Implement search result navigation
    - Click result to open player at timestamp
    - _Requirements: 3.4_
  
  - [ ] 22.2 Implement topic search trigger
    - Click topic to perform search
    - _Requirements: 9.4_
  
  - [ ] 22.3 Implement visual feedback for actions
    - Loading spinners
    - Progress bars
    - Success/error notifications
    - _Requirements: 8.2_
  
  - [ ] 22.4 Implement error message display
    - User-friendly error messages
    - Actionable guidance
    - _Requirements: 8.3_
  
  - [ ]* 22.5 Write property test for search result navigation
    - **Property 27: Search result navigation**
    - **Validates: Requirements 3.4**
  
  - [ ]* 22.6 Write property test for topic search trigger
    - **Property 28: Topic search trigger**
    - **Validates: Requirements 9.4**

- [ ] 23. Implement keyboard shortcuts
  - [ ] 23.1 Define keyboard shortcuts
    - Search: Cmd/Ctrl + F
    - Play/Pause: Space
    - Next scene: Cmd/Ctrl + Right
    - Previous scene: Cmd/Ctrl + Left
    - _Requirements: 8.5_
  
  - [ ] 23.2 Implement shortcut handlers
    - Register global shortcuts
    - Handle shortcuts in player view
    - _Requirements: 8.5_
  
  - [ ]* 23.3 Write unit tests for keyboard shortcuts
    - Test shortcut registration
    - Test shortcut handlers
    - _Requirements: 8.5_

- [ ] 24. Implement packaging and distribution
  - [ ] 24.1 Configure electron-builder
    - Set up build configuration for Windows, macOS, Linux
    - Configure app icons and metadata
    - _Requirements: 8.1_
  
  - [ ] 24.2 Create installers
    - Build .exe for Windows
    - Build .dmg for macOS
    - Build .AppImage for Linux
    - _Requirements: 8.1_
  
  - [ ] 24.3 Test installers on target platforms
    - Test installation and uninstallation
    - Test app launch and basic functionality
    - _Requirements: 8.1_

- [ ] 25. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 26. Implement action detection pipeline
  - [ ] 26.1 Implement action detection service
    - Load X3D, SlowFast, or TimeSformer model for action recognition
    - Sample video clips at configured interval (window size + stride)
    - Run action detection on sampled clips
    - Handle GPU acceleration if available
    - Use Kinetics-400 labels for action classification
    - _Requirements: 6.6_
  
  - [ ] 26.2 Implement action detection task handler
    - Create ActionDetectionTaskHandler following existing pattern
    - Integrate with task orchestration system
    - Handle task lifecycle (pending → running → completed/failed)
    - Store action detections in database
    - _Requirements: 6.2, 6.5_
  
  - [ ] 26.3 Create Actions database table and DAO
    - Define columns: actionId, videoId, label, timestamps (JSON), confidence, metadata (JSON)
    - Add foreign key to Videos table
    - Add index on videoId and label
    - Implement CRUD operations in Action DAO
    - _Requirements: 4.1, 4.2_
  
  - [ ] 26.4 Integrate action detection worker
    - Add ActionDetectionWorker to worker pool manager
    - Configure worker count and priority in processing profiles
    - Add ACTION_DETECTION to TaskType enum
    - _Requirements: 6.2_
  
  - [ ] 26.5 Implement action detection API endpoint
    - GET /v1/videos/{id}/actions - get detected actions
    - Return action labels, timestamps, and confidence scores
    - _Requirements: 8.1, 8.2_
  
  - [ ]* 26.6 Write unit tests for action detection
    - Test clip sampling and window generation
    - Test action detection aggregation
    - Test GPU fallback to CPU
    - Test API endpoint
    - _Requirements: 6.6_

- [ ] 27. Implement emotion detection pipeline
  - [ ] 27.1 Implement emotion detection service
    - Load DeepFace library for emotion analysis
    - Sample frames at configured interval
    - Detect faces and analyze emotions per face
    - Handle GPU acceleration if available
    - Store dominant emotion and confidence scores
    - _Requirements: 6.6_
  
  - [ ] 27.2 Implement emotion detection task handler
    - Create EmotionDetectionTaskHandler following existing pattern
    - Integrate with task orchestration system
    - Handle task lifecycle (pending → running → completed/failed)
    - Store emotion detections in database
    - _Requirements: 6.2, 6.5_
  
  - [ ] 27.3 Create Emotions database table and DAO
    - Define columns: emotionId, videoId, faceId, timestamps (JSON), dominantEmotion, emotionScores (JSON), confidence
    - Add foreign key to Videos table
    - Add index on videoId and dominantEmotion
    - Implement CRUD operations in Emotion DAO
    - _Requirements: 4.1, 4.2_
  
  - [ ] 27.4 Integrate emotion detection worker
    - Add EmotionDetectionWorker to worker pool manager
    - Configure worker count and priority in processing profiles
    - Add EMOTION_DETECTION to TaskType enum
    - _Requirements: 6.2_
  
  - [ ] 27.5 Implement emotion detection API endpoint
    - GET /v1/videos/{id}/emotions - get detected emotions
    - Return emotion labels, timestamps, face associations, and scores
    - _Requirements: 8.1, 8.2_
  
  - [ ]* 27.6 Write unit tests for emotion detection
    - Test frame sampling
    - Test emotion detection per face
    - Test GPU fallback to CPU
    - Test API endpoint
    - _Requirements: 6.6_

- [ ] 28. Implement places detection pipeline
  - [ ] 28.1 Implement places detection service
    - Load ResNet18 Places365 model for scene classification
    - Sample frames at configured interval
    - Classify scenes/locations using Places365 categories
    - Handle GPU acceleration if available
    - Store top-5 place predictions with confidence scores
    - _Requirements: 6.6_
  
  - [ ] 28.2 Implement places detection task handler
    - Create PlacesDetectionTaskHandler following existing pattern
    - Integrate with task orchestration system
    - Handle task lifecycle (pending → running → completed/failed)
    - Store place detections in database
    - _Requirements: 6.2, 6.5_
  
  - [ ] 28.3 Create Places database table and DAO
    - Define columns: placeId, videoId, label, timestamps (JSON), confidence, alternativeLabels (JSON)
    - Add foreign key to Videos table
    - Add index on videoId and label
    - Implement CRUD operations in Place DAO
    - _Requirements: 4.1, 4.2_
  
  - [ ] 28.4 Integrate places detection worker
    - Add PlacesDetectionWorker to worker pool manager
    - Configure worker count and priority in processing profiles
    - Add PLACES_DETECTION to TaskType enum
    - _Requirements: 6.2_
  
  - [ ] 28.5 Implement places detection API endpoint
    - GET /v1/videos/{id}/places - get detected places/scenes
    - Return place labels, timestamps, and confidence scores
    - _Requirements: 8.1, 8.2_
  
  - [ ]* 28.6 Write unit tests for places detection
    - Test frame sampling
    - Test scene classification
    - Test GPU fallback to CPU
    - Test API endpoint
    - _Requirements: 6.6_

- [ ] 29. Implement OCR text detection pipeline
  - [ ] 29.1 Implement OCR text detection service
    - Load EasyOCR library for text extraction
    - Sample frames at configured interval (or time-based)
    - Extract text with bounding boxes and confidence scores
    - Support include/exclude regions for targeted text extraction
    - Handle GPU acceleration if available (MPS, CUDA, or CPU)
    - _Requirements: 6.6_
  
  - [ ] 29.2 Implement OCR text detection task handler
    - Create OcrTextDetectionTaskHandler following existing pattern
    - Integrate with task orchestration system
    - Handle task lifecycle (pending → running → completed/failed)
    - Store text detections in database
    - _Requirements: 6.2, 6.5_
  
  - [ ] 29.3 Create OcrTexts database table and DAO
    - Define columns: textId, videoId, text, timestamps (JSON), boundingBoxes (JSON), confidence, metadata (JSON)
    - Add foreign key to Videos table
    - Add index on videoId and text (full-text search)
    - Implement CRUD operations in OcrText DAO
    - _Requirements: 4.1, 4.2_
  
  - [ ] 29.4 Integrate OCR text detection worker
    - Add OcrTextDetectionWorker to worker pool manager
    - Configure worker count and priority in processing profiles
    - Add OCR_TEXT_DETECTION to TaskType enum
    - _Requirements: 6.2_
  
  - [ ] 29.5 Implement OCR text detection API endpoint
    - GET /v1/videos/{id}/ocr-text - get detected text
    - Return text content, timestamps, bounding boxes, and confidence
    - Support search/filter by text content
    - _Requirements: 8.1, 8.2_
  
  - [ ]* 29.6 Write unit tests for OCR text detection
    - Test frame sampling (frame-based and time-based)
    - Test text extraction with bounding boxes
    - Test include/exclude regions
    - Test GPU fallback to CPU
    - Test API endpoint
    - _Requirements: 6.6_

- [ ] 30. Checkpoint - Ensure all detection pipeline tests pass
  - Ensure all tests pass for new detection types
  - Verify worker pool integration
  - Validate API endpoints
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The implementation follows a bottom-up approach: data layer → processing → API → UI
- GPU acceleration is optional but recommended for faster processing
- All file operations use relative paths for portability
