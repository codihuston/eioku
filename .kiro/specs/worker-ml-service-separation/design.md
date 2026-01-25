# Design Document: Worker & ML Service Separation

## Overview

This document describes the architectural design for separating the monolithic Eioku backend into three independent services: API Service, Worker Service, and ML Service. The design enables independent scaling, resource isolation, and cleaner deployment boundaries while maintaining backward compatibility with existing REST APIs and workflows.

### Key Design Principles

1. **PostgreSQL as Source of Truth**: All persistent state lives in PostgreSQL; Redis is ephemeral
2. **Time-Free Reconciliation**: Reconciler uses definitive Redis signals, never time-based thresholds
3. **Shared Queue Pattern**: Both Worker and ML Service consume from same Redis queues (no HTTP calls)
4. **Result Polling**: Worker polls PostgreSQL for artifact completion instead of blocking on HTTP
5. **Async-First Architecture**: All I/O operations use async/await for efficiency
6. **Graceful Degradation**: Services can fail independently without cascading failures
7. **Horizontal Scalability**: All services can scale independently via Docker Compose or Kubernetes

## Architecture

### High-Level Service Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                         External Clients                         │
│                      (Web, CLI, Mobile)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Service (Port 8000)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FastAPI Server                                           │   │
│  │ - REST endpoints (videos, tasks, artifacts, paths)      │   │
│  │ - Task orchestration & job enqueueing                   │   │
│  │ - Artifact queries & navigation                         │   │
│  │ - Selection policies & projections                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                    ┌────────┴────────┐                           │
│                    ▼                 ▼                           │
│            PostgreSQL         Redis (arq)                        │
│            Connection         Job Producer                       │
│            Pool (20)          (gpu_jobs, cpu_jobs)              │
└─────────────────────────────────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  PostgreSQL      │  │  Redis/Valkey    │
        │  (Source of      │  │  (Job Queues)    │
        │   Truth)         │  │  (Ephemeral)     │
        │                  │  │  - gpu_jobs      │
        │  - tasks         │  │  - cpu_jobs      │
        │  - artifacts     │  │  - ml_jobs       │
        └──────────────────┘  └──────────────────┘
                    ▲                 │
                    │                 ├─────────────────┐
                    │                 ▼                 ▼
        ┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
        │    Worker Service (No HTTP)          │  │    ML Service (No HTTP)              │
        │  ┌──────────────────────────────────┐│  │  ┌──────────────────────────────────┐│
        │  │ arq Consumer                     ││  │  │ arq Consumer                     ││
        │  │ - Consume from gpu_jobs/cpu_jobs││  │  │ - Consume from ml_jobs queue     ││
        │  │ - Enqueue to ml_jobs queue       ││  │  │ - Execute ML inference           ││
        │  │ - Poll PostgreSQL for results    ││  │  │ - Create ArtifactEnvelopes       ││
        │  │ - Update task status             ││  │  │ - Batch insert to PostgreSQL     ││
        │  │ - Reconciliation (every 5 min)   ││  │  │ - Acknowledge job in Redis       ││
        │  └──────────────────────────────────┘│  │  └──────────────────────────────────┘│
        │                                      │  │                                      │
        │  Connection Pool (10)                │  │  Connection Pool (10)                │
        │  Polling with exponential backoff    │  │  GPU Semaphore (concurrency=2)       │
        └──────────────────────────────────────┘  │  Model Cache (lazy-loaded)           │
                                                   └──────────────────────────────────────┘
```

### Service Responsibilities

| Service | Responsibilities | Scaling | Deployment |
|---------|------------------|---------|------------|
| **API Service** | REST endpoints, task orchestration, job enqueueing to gpu_jobs/cpu_jobs, artifact queries | Horizontal (stateless) | Separate container |
| **Worker Service** | Consume from gpu_jobs/cpu_jobs, enqueue to ml_jobs, poll PostgreSQL for results, reconciliation | Horizontal (stateless) | Separate container |
| **ML Service** | Consume from ml_jobs, execute ML inference, persist artifacts to PostgreSQL, GPU management | Horizontal (GPU-aware) | Separate container |

### Data Flow Diagrams

#### Task Creation Flow (Automatic)

```
Video Discovery Service
    │
    ├─ Scan filesystem for new videos
    ├─ Check if video already in PostgreSQL
    │
    ▼
API Service: Video discovered
    │
    ├─ Create video record (status=DISCOVERED)
    ├─ Auto-create task records for each task type
    │  (object_detection, face_detection, transcription, ocr, place_detection, scene_detection)
    ├─ Determine GPU requirement for each task type
    │  (object_detection, face_detection, place_detection, scene_detection → GPU)
    │  (transcription, ocr → CPU or GPU)
    ├─ Enqueue jobs to Redis (gpu_jobs or cpu_jobs queue)
    │
    ▼
PostgreSQL: INSERT videos, INSERT tasks (status=PENDING)
    │
    ▼
Redis: XADD gpu_jobs or cpu_jobs {task_id, task_type, video_id, video_path}
    │
    ▼
Worker Service (GPU or CPU mode): Begins consuming jobs from appropriate queue
```

#### Task Creation Flow (Manual - Optional)

```
Client Request
    │
    ▼
API Service: POST /tasks/{task_id}/enqueue
    │
    ├─ Check if task exists and is PENDING
    ├─ Enqueue job to Redis
    │
    ▼
Redis: XADD ml_jobs {task_id, task_type, video_id, video_path}
    │
    ▼
Response: {task_id, job_id, status}
```

#### Job Execution Flow (Shared Queue Pattern)

```
Worker Service: arq Consumer (gpu_jobs or cpu_jobs)
    │
    ├─ XREADGROUP from gpu_jobs or cpu_jobs
    │
    ▼
Redis: Job payload {task_id, task_type, video_id, video_path, config}
    │
    ├─ Check task status (pre-flight)
    ├─ Update task status to RUNNING in PostgreSQL
    │
    ▼
PostgreSQL: UPDATE tasks SET status=RUNNING, started_at=NOW()
    │
    ▼
Worker Service: Enqueue to ml_jobs queue
    │
    ├─ XADD ml_jobs {task_id, task_type, video_id, video_path, config}
    │
    ▼
Redis: ml_jobs queue now has job
    │
    ├─ Worker Service begins polling PostgreSQL for artifacts
    ├─ Poll with exponential backoff (1s, 2s, 4s, 8s, etc.)
    │
    ▼
ML Service: arq Consumer (ml_jobs)
    │
    ├─ XREADGROUP from ml_jobs
    │
    ▼
Redis: Job payload
    │
    ├─ Load model (lazy, with GPU semaphore)
    ├─ Process video
    ├─ Create ArtifactEnvelopes with provenance
    │
    ▼
ML Service: Batch insert artifacts to PostgreSQL
    │
    ├─ INSERT INTO artifacts (task_id, artifact_type, payload_json, config_hash, input_hash, ...)
    ├─ XACK job in Redis
    │
    ▼
PostgreSQL: Artifacts inserted, projections updated via triggers
    │
    ▼
Worker Service: Polling detects artifacts
    │
    ├─ SELECT COUNT(*) FROM artifacts WHERE task_id = ?
    ├─ Verify all expected artifacts present
    ├─ Update task status to COMPLETED
    ├─ XACK job in Redis (gpu_jobs or cpu_jobs)
    │
    ▼
PostgreSQL: Task marked COMPLETED, started_at and completed_at recorded
```

#### Reconciliation Flow

```
Reconciler (every 5 minutes)
    │
    ├─ Check PENDING tasks
    │  ├─ For each: Is job in Redis?
    │  ├─ If NO: Re-enqueue (handles Redis data loss)
    │  └─ If YES: Do nothing (arq will handle)
    │
    ├─ Check RUNNING tasks
    │  ├─ For each: Get job status from Redis
    │  ├─ If NOT FOUND: Reset to PENDING, re-enqueue
    │  ├─ If COMPLETE: Sync to PostgreSQL
    │  ├─ If FAILED: Sync to PostgreSQL with error
    │  └─ If IN_PROGRESS: Do nothing
    │
    └─ Check long-running tasks
       ├─ For each: Is running > threshold?
       ├─ If YES: Send alert (never auto-kill)
       └─ If NO: Do nothing
```

## Components and Interfaces

### API Service Components

#### FastAPI Application

```python
# backend/src/main_api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Eioku API Service")

# Middleware
app.add_middleware(CORSMiddleware, ...)

# Routers
app.include_router(video_router)
app.include_router(task_router)
app.include_router(artifact_router)
app.include_router(path_router)
app.include_router(selection_router)

# Startup/shutdown
@app.on_event("startup")
async def startup():
    # Initialize database pool
    # Initialize Redis connection for job enqueueing
    pass

@app.on_event("shutdown")
async def shutdown():
    # Close connections gracefully
    pass
```

#### Video Discovery Service

```python
# backend/src/services/video_discovery_service.py
class VideoDiscoveryService:
    def __init__(self, db: Database, job_producer: JobProducer):
        self.db = db
        self.job_producer = job_producer
    
    async def discover_and_queue_tasks(self, video_path: str):
        """Discover video and auto-create tasks for all ML operations"""
        # Check if video already exists
        existing = await self.db.fetchrow(
            "SELECT video_id FROM videos WHERE file_path = :path",
            {"path": video_path}
        )
        
        if existing:
            return existing['video_id']
        
        # Create video record
        video_id = str(uuid4())
        file_hash = compute_file_hash(video_path)
        duration = get_video_duration(video_path)
        
        await self.db.execute(
            "INSERT INTO videos (video_id, file_path, file_hash, duration, status) "
            "VALUES (:id, :path, :hash, :duration, 'DISCOVERED')",
            {
                "id": video_id,
                "path": video_path,
                "hash": file_hash,
                "duration": duration
            }
        )
        
        # Auto-create tasks for all ML operations
        task_types = [
            'object_detection',
            'face_detection',
            'transcription',
            'ocr',
            'place_detection',
            'scene_detection'
        ]
        
        for task_type in task_types:
            task_id = str(uuid4())
            
            # Create task record
            await self.db.execute(
                "INSERT INTO tasks (task_id, video_id, task_type, status, created_at) "
                "VALUES (:task_id, :video_id, :task_type, 'PENDING', NOW())",
                {
                    "task_id": task_id,
                    "video_id": video_id,
                    "task_type": task_type
                }
            )
            
            # Enqueue job to Redis
            config = self._get_default_config(task_type)
            await self.job_producer.enqueue_task(
                task_id=task_id,
                task_type=task_type,
                video_id=video_id,
                video_path=video_path,
                config=config
            )
        
        return video_id
    
    def _get_default_config(self, task_type: str) -> dict:
        """Get default configuration for task type"""
        configs = {
            'object_detection': {
                'model_name': 'yolov8n.pt',
                'frame_interval': 30,
                'confidence_threshold': 0.5,
                'model_profile': 'balanced'
            },
            'face_detection': {
                'model_name': 'yolov8n-face.pt',
                'frame_interval': 30,
                'confidence_threshold': 0.5
            },
            'transcription': {
                'model_name': 'large-v3',
                'language': None,
                'vad_filter': True
            },
            'ocr': {
                'frame_interval': 60,
                'languages': ['en'],
                'use_gpu': True
            },
            'place_detection': {
                'frame_interval': 60,
                'top_k': 5
            },
            'scene_detection': {
                'threshold': 0.4,
                'min_scene_length': 0.6
            }
        }
        return configs.get(task_type, {})
```

#### Job Producer with GPU/CPU Routing

```python
# backend/src/services/job_producer.py
class JobProducer:
    def __init__(self, redis_settings: RedisSettings):
        self.redis_settings = redis_settings
        
        # Task types that require GPU
        self.gpu_tasks = {
            'object_detection',
            'face_detection',
            'place_detection',
            'scene_detection'
        }
        
        # Task types that can run on CPU or GPU
        self.flexible_tasks = {
            'transcription',
            'ocr'
        }
    
    def _get_queue_name(self, task_type: str) -> str:
        """Determine which queue to use based on task type"""
        if task_type in self.gpu_tasks:
            return 'gpu_jobs'
        elif task_type in self.flexible_tasks:
            # Could be routed to either queue based on config
            # For now, default to gpu_jobs if available
            return 'gpu_jobs'
        else:
            return 'cpu_jobs'
    
    async def enqueue_task(self, task_id: str, task_type: str, video_id: str, 
                          video_path: str, config: dict):
        """Enqueue a task to Redis via arq to appropriate queue"""
        redis = await create_pool(self.redis_settings)
        queue_name = self._get_queue_name(task_type)
        
        await redis.enqueue_job(
            'process_ml_task',
            task_id=task_id,
            task_type=task_type,
            video_id=video_id,
            video_path=video_path,
            config=config,
            _job_id=f"ml_{task_id}",
            _queue_name=queue_name
        )
        
        logger.info(f"Enqueued task {task_id} ({task_type}) to {queue_name}")
```

#### Manual Task Enqueueing Endpoint

```python
# backend/src/api/task_routes.py
@router.post("/tasks/{task_id}/enqueue")
async def enqueue_task(task_id: str, db: Database, job_producer: JobProducer):
    """Manually enqueue a task (for user-triggered processing)"""
    # Get task details
    task = await db.fetchrow(
        "SELECT * FROM tasks WHERE task_id = :id",
        {"id": task_id}
    )
    
    if not task:
        raise HTTPException(404, "Task not found")
    
    if task['status'] != 'PENDING':
        raise HTTPException(400, f"Cannot enqueue task in status {task['status']}")
    
    # Get video details
    video = await db.fetchrow(
        "SELECT file_path FROM videos WHERE video_id = :id",
        {"id": task['video_id']}
    )
    
    # Get default config for task type
    discovery_service = VideoDiscoveryService(db, job_producer)
    config = discovery_service._get_default_config(task['task_type'])
    
    # Enqueue to Redis
    await job_producer.enqueue_task(
        task_id=task_id,
        task_type=task['task_type'],
        video_id=str(task['video_id']),
        video_path=video['file_path'],
        config=config
    )
    
    return {
        "task_id": task_id,
        "job_id": f"ml_{task_id}",
        "status": "enqueued"
    }
```

### Worker Service Components

#### arq Worker Configuration with GPU/CPU Mode

```python
# backend/src/workers/arq_worker.py
from arq import cron
from arq.connections import RedisSettings
import torch

# Determine GPU mode from environment
GPU_MODE = os.getenv('GPU_MODE', 'auto')  # gpu, cpu, or auto

def get_queue_names():
    """Determine which queues this worker should consume from"""
    if GPU_MODE == 'gpu':
        return ['gpu_jobs']
    elif GPU_MODE == 'cpu':
        return ['cpu_jobs']
    elif GPU_MODE == 'auto':
        # Auto-detect GPU availability
        if torch.cuda.is_available():
            logger.info("GPU detected - consuming from gpu_jobs queue")
            return ['gpu_jobs']
        else:
            logger.info("GPU not available - consuming from cpu_jobs queue")
            return ['cpu_jobs']
    else:
        raise ValueError(f"Invalid GPU_MODE: {GPU_MODE}")

class WorkerSettings:
    functions = [process_ml_task]
    redis_settings = RedisSettings(host='redis', port=6379)
    max_jobs = int(os.getenv('ARQ_MAX_JOBS', 4))
    job_timeout = int(os.getenv('ARQ_JOB_TIMEOUT', 1800))
    max_tries = 3
    allow_abort_jobs = True
    queue_name = get_queue_names()  # Consume from appropriate queue(s)
    
    @cron('*/5 * * * *')  # Every 5 minutes
    async def reconcile_tasks(ctx):
        """Periodic reconciliation task"""
        reconciler = ctx['reconciler']
        await reconciler.run()
```

#### Task Handler

```python
# backend/src/workers/task_handlers.py
async def process_ml_task(ctx, task_id: str, task_type: str, video_id: str, 
                          video_path: str, config: dict):
    """Main task handler - dispatches to ML Service"""
    db = ctx['db']
    ml_client = ctx['ml_client']
    
    # Pre-flight check
    task = await db.fetchrow("SELECT status FROM tasks WHERE task_id = :id", 
                             {"id": task_id})
    if task.status in ('COMPLETED', 'CANCELLED'):
        return {"status": "skipped", "reason": task.status}
    
    # Mark running
    await db.execute("UPDATE tasks SET status = 'RUNNING', started_at = NOW() "
                     "WHERE task_id = :id", {"id": task_id})
    
    try:
        # Dispatch to ML Service
        result = await ml_client.infer(task_type, video_path, config)
        
        # Transform and persist
        artifacts = transform_to_envelopes(result, task_id, video_id)
        await persist_artifacts(db, artifacts)
        
        # Mark complete
        await db.execute("UPDATE tasks SET status = 'COMPLETED', "
                         "completed_at = NOW() WHERE task_id = :id",
                         {"id": task_id})
        
        return {"status": "completed", "artifact_count": len(artifacts)}
    
    except asyncio.CancelledError:
        await db.execute("UPDATE tasks SET status = 'CANCELLED', "
                         "error = 'Aborted by user' WHERE task_id = :id",
                         {"id": task_id})
        raise
    
    except Exception as e:
        await db.execute("UPDATE tasks SET status = 'FAILED', error = :error "
                         "WHERE task_id = :id", {"id": task_id, "error": str(e)})
        raise
```

#### ML Client

```python
# backend/src/services/ml_client.py
class MLClient:
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=600)
    
    async def infer(self, task_type: str, video_path: str, config: dict):
        """Call ML Service endpoint"""
        endpoint_map = {
            'object_detection': '/infer/objects',
            'face_detection': '/infer/faces',
            'transcription': '/infer/transcribe',
            'ocr': '/infer/ocr',
            'place_detection': '/infer/places',
            'scene_detection': '/infer/scenes'
        }
        
        endpoint = endpoint_map[task_type]
        response = await self.client.post(endpoint, json={
            "video_path": video_path,
            **config
        })
        response.raise_for_status()
        return response.json()
```

#### Reconciler

```python
# backend/src/workers/reconciler.py
class Reconciler:
    def __init__(self, db: Database, redis: ArqRedis):
        self.db = db
        self.redis = redis
    
    async def run(self):
        """Run all reconciliation checks"""
        await self._sync_pending_tasks()
        await self._sync_running_tasks()
        await self._alert_long_running_tasks()
    
    async def _sync_pending_tasks(self):
        """Ensure all PENDING tasks have jobs in Redis"""
        pending = await self.db.fetch(
            "SELECT task_id FROM tasks WHERE status = 'PENDING'"
        )
        for task in pending:
            job_info = await self._get_job_info(task.task_id)
            if job_info is None:
                logger.info(f"Re-enqueueing missing job for {task.task_id}")
                await self._enqueue(task)
    
    async def _sync_running_tasks(self):
        """Sync RUNNING tasks with Redis state"""
        running = await self.db.fetch(
            "SELECT task_id FROM tasks WHERE status = 'RUNNING'"
        )
        for task in running:
            job_info = await self._get_job_info(task.task_id)
            if job_info is None:
                # Job vanished - re-enqueue
                await self.db.execute(
                    "UPDATE tasks SET status = 'PENDING' WHERE task_id = :id",
                    {"id": task.task_id}
                )
                await self._enqueue(task)
            elif job_info.status == "complete":
                await self.db.execute(
                    "UPDATE tasks SET status = 'COMPLETED' WHERE task_id = :id",
                    {"id": task.task_id}
                )
            elif job_info.status == "failed":
                await self.db.execute(
                    "UPDATE tasks SET status = 'FAILED' WHERE task_id = :id",
                    {"id": task.task_id}
                )
```

### ML Service Components

#### FastAPI Application with Model Initialization

```python
# ml-service/src/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch

# GPU semaphore for concurrency control
GPU_SEMAPHORE = asyncio.Semaphore(int(os.getenv('GPU_CONCURRENCY', 2)))

# Global model registry
MODELS = {}
GPU_AVAILABLE = False
INITIALIZATION_ERROR = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Download and verify all models
    global GPU_AVAILABLE, INITIALIZATION_ERROR
    
    try:
        # Check GPU availability
        GPU_AVAILABLE = torch.cuda.is_available()
        require_gpu = os.getenv('REQUIRE_GPU', 'false').lower() == 'true'
        
        logger.info(f"GPU available: {GPU_AVAILABLE}")
        
        if GPU_AVAILABLE:
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            if require_gpu:
                raise RuntimeError("GPU required but not available (set REQUIRE_GPU=false to allow CPU-only mode)")
            else:
                logger.warning("GPU not available - will use CPU for inference (slower)")
        
        # Download and verify all models
        await initialize_all_models()
        logger.info("All models initialized successfully")
        
    except Exception as e:
        INITIALIZATION_ERROR = str(e)
        logger.error(f"Model initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown: Unload models and cleanup
    await cleanup_all_models()

app = FastAPI(title="Eioku ML Service", lifespan=lifespan)

# Routers
app.include_router(object_detection_router)
app.include_router(face_detection_router)
app.include_router(transcription_router)
app.include_router(ocr_router)
app.include_router(place_detection_router)
app.include_router(scene_detection_router)
app.include_router(health_router)

async def initialize_all_models():
    """Download and verify all models on startup"""
    model_configs = {
        'object_detection': {
            'model_name': 'yolov8n.pt',
            'service': ObjectDetectionService
        },
        'face_detection': {
            'model_name': 'yolov8n-face.pt',
            'service': FaceDetectionService
        },
        'transcription': {
            'model_name': 'large-v3',
            'service': TranscriptionService
        },
        'ocr': {
            'model_name': 'easyocr',
            'service': OCRService
        },
        'place_detection': {
            'model_name': 'resnet18_places365',
            'service': PlaceDetectionService
        },
        'scene_detection': {
            'model_name': 'scenedetect',
            'service': SceneDetectionService
        }
    }
    
    for task_type, config in model_configs.items():
        logger.info(f"Initializing {task_type} model: {config['model_name']}")
        
        try:
            service = config['service']()
            
            # Download model (will be cached after first download)
            await service.download_model(config['model_name'])
            
            # Verify model loads and GPU detection works
            await service.verify_model(config['model_name'])
            
            # Store in registry
            MODELS[task_type] = {
                'service': service,
                'model_name': config['model_name'],
                'status': 'ready'
            }
            
            logger.info(f"✓ {task_type} model initialized successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize {task_type}: {e}")
            MODELS[task_type] = {
                'status': 'failed',
                'error': str(e)
            }
            raise

async def cleanup_all_models():
    """Unload models and cleanup resources"""
    for task_type, model_info in MODELS.items():
        if model_info['status'] == 'ready':
            try:
                await model_info['service'].cleanup()
                logger.info(f"Cleaned up {task_type} model")
            except Exception as e:
                logger.error(f"Error cleaning up {task_type}: {e}")
```

#### Model Initialization Services

```python
# ml-service/src/services/model_manager.py
class ModelManager:
    """Manages model download, verification, and lifecycle"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.models = {}
    
    async def download_model(self, model_name: str, model_type: str):
        """Download model from source and cache locally"""
        cache_path = os.path.join(self.cache_dir, model_name)
        
        if os.path.exists(cache_path):
            logger.info(f"Model {model_name} already cached at {cache_path}")
            return cache_path
        
        logger.info(f"Downloading {model_type} model: {model_name}")
        
        if model_type == 'yolo':
            from ultralytics import YOLO
            model = YOLO(model_name)
            model.save(cache_path)
        elif model_type == 'whisper':
            from faster_whisper import WhisperModel
            model = WhisperModel(model_name, device="auto", compute_type="auto")
            # faster_whisper handles caching internally
        elif model_type == 'places365':
            # Download ResNet18 Places365 model
            download_places365_model(cache_path)
        
        logger.info(f"✓ Downloaded {model_name} to {cache_path}")
        return cache_path
    
    async def verify_model(self, model_name: str, model_type: str):
        """Verify model loads and GPU detection works"""
        logger.info(f"Verifying {model_type} model: {model_name}")
        
        try:
            if model_type == 'yolo':
                from ultralytics import YOLO
                model = YOLO(model_name)
                # Test on dummy image
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                results = model(dummy_image, verbose=False)
                logger.info(f"✓ YOLO model {model_name} verified")
                
            elif model_type == 'whisper':
                from faster_whisper import WhisperModel
                model = WhisperModel(model_name, device="auto", compute_type="auto")
                logger.info(f"✓ Whisper model {model_name} verified")
                
            elif model_type == 'places365':
                # Verify ResNet18 model loads
                model = load_places365_model(model_name)
                logger.info(f"✓ Places365 model {model_name} verified")
            
            # Log GPU detection result
            if torch.cuda.is_available():
                logger.info(f"  GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.info(f"  GPU not available, using CPU")
                
        except Exception as e:
            logger.error(f"✗ Model verification failed for {model_name}: {e}")
            raise
```

#### Inference Endpoints

```python
# ml-service/src/api/object_detection.py
from fastapi import APIRouter
from ml_service.models.requests import ObjectDetectionRequest
from ml_service.models.responses import ObjectDetectionResponse

router = APIRouter(prefix="/infer", tags=["inference"])

@router.post("/objects", response_model=ObjectDetectionResponse)
async def detect_objects(request: ObjectDetectionRequest):
    """Detect objects in video"""
    # Models are already loaded at startup, just use them
    async with GPU_SEMAPHORE:
        service = MODELS['object_detection']['service']
        detections = await service.detect(
            video_path=request.video_path,
            model_name=request.model_name,
            frame_interval=request.frame_interval,
            confidence_threshold=request.confidence_threshold,
            model_profile=request.model_profile
        )
        
        return ObjectDetectionResponse(
            run_id=str(uuid4()),
            config_hash=compute_config_hash(request),
            input_hash=compute_input_hash(request.video_path),
            model_profile=request.model_profile,
            producer="yolo",
            producer_version="8.0.0",
            detections=detections
        )
```

#### Response Models

```python
# ml-service/src/models/responses.py
from pydantic import BaseModel

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Detection(BaseModel):
    frame_index: int
    timestamp_ms: int
    label: str
    confidence: float
    bbox: BoundingBox

class ObjectDetectionResponse(BaseModel):
    run_id: str
    config_hash: str
    input_hash: str
    model_profile: str
    producer: str
    producer_version: str
    detections: list[Detection]
```

### Artifact Transformation

#### Envelope Creation

```python
# backend/src/domain/artifacts.py
def transform_to_envelopes(ml_response: dict, task_id: str, video_id: str):
    """Transform ML response to ArtifactEnvelopes"""
    envelopes = []
    
    for detection in ml_response['detections']:
        envelope = ArtifactEnvelope(
            artifact_id=str(uuid4()),
            asset_id=video_id,
            artifact_type="object.detection",
            schema_version=1,
            span_start_ms=detection['timestamp_ms'],
            span_end_ms=detection['timestamp_ms'],
            payload_json={
                "label": detection['label'],
                "confidence": detection['confidence'],
                "bounding_box": detection['bbox']
            },
            producer=ml_response['producer'],
            producer_version=ml_response['producer_version'],
            model_profile=ml_response['model_profile'],
            config_hash=ml_response['config_hash'],
            input_hash=ml_response['input_hash'],
            run_id=ml_response['run_id']
        )
        envelopes.append(envelope)
    
    return envelopes
```

#### Schema Validation

```python
# backend/src/domain/schemas/object_detection_v1.py
from pydantic import BaseModel

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class ObjectDetectionV1(BaseModel):
    """Validates payload_json for object detection artifacts"""
    label: str
    confidence: float
    bounding_box: BoundingBox
```

## Data Models

### Task Status Lifecycle

```
PENDING → RUNNING → COMPLETED
   ↓        ↓
   └─→ FAILED
   
PENDING → CANCELLED (via API)
RUNNING → CANCELLED (via API)
FAILED → PENDING (via retry API)
CANCELLED → PENDING (via retry API)
```

### Database Schema (No Changes Required)

The existing schema supports the new architecture:

```sql
-- Existing tables (unchanged)
CREATE TABLE tasks (
    task_id UUID PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos(video_id),
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,  -- PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    priority INT DEFAULT 0,
    dependencies JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE artifacts (
    artifact_id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES videos(video_id),
    artifact_type VARCHAR(100) NOT NULL,
    schema_version INT NOT NULL,
    span_start_ms INT,
    span_end_ms INT,
    payload_json JSONB NOT NULL,
    producer VARCHAR(50),
    model_profile VARCHAR(50),
    config_hash VARCHAR(64),
    input_hash VARCHAR(64),
    run_id UUID,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for reconciliation performance
CREATE INDEX idx_tasks_pending_age ON tasks(created_at) 
WHERE status = 'PENDING';

CREATE INDEX idx_tasks_running_age ON tasks(started_at) 
WHERE status = 'RUNNING';
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property-Based Testing Overview

Property-based testing (PBT) validates software correctness by testing universal properties across many generated inputs. Each property is a formal specification that should hold for all valid inputs.

#### Core Principles

1. **Universal Quantification**: Every property must contain an explicit "for all" statement
2. **Requirements Traceability**: Each property must reference the requirements it validates
3. **Executable Specifications**: Properties must be implementable as automated tests
4. **Comprehensive Coverage**: Properties should cover all testable acceptance criteria

### Acceptance Criteria Testing Prework



#### Prework Analysis

1.1 WHEN a video is discovered, THE API_Service SHALL create task records in PostgreSQL
  Thoughts: This is a rule that should apply to all videos. We can generate random video paths, trigger discovery, then verify that 6 task records are created (one for each task type).
  Testable: yes - property

1.2 WHEN a task is created, THE API_Service SHALL enqueue a job to Redis via arq
  Thoughts: This is a rule that should apply to all tasks. We can create a task and verify that a corresponding job exists in Redis.
  Testable: yes - property

2.1 WHEN a job is consumed, THE Worker_Service SHALL update the corresponding task status to RUNNING in PostgreSQL
  Thoughts: This is a rule that should apply to all jobs. We can enqueue a job, let the worker consume it, then verify the task status is RUNNING.
  Testable: yes - property

2.2 WHEN the Worker_Service completes a job, THE Worker_Service SHALL acknowledge the job in Redis (XACK)
  Thoughts: This is a rule that should apply to all completed jobs. We can verify that acknowledged jobs are removed from the pending list.
  Testable: yes - property

3.1 WHEN an inference endpoint receives a request, THE ML_Service SHALL return structured results including detections/classifications and provenance metadata
  Thoughts: This is a rule that should apply to all inference requests. We can generate random video paths and verify the response contains required fields.
  Testable: yes - property

4.1 WHEN a job is enqueued, THE API_Service SHALL assign a unique job_id in format "ml_{task_id}" for deduplication
  Thoughts: This is a rule that should apply to all enqueued jobs. We can verify that job_id follows the format and is unique.
  Testable: yes - property

5.1 WHEN a task is created, THE task status SHALL be PENDING in PostgreSQL
  Thoughts: This is a rule that should apply to all newly created tasks. We can create a task and verify its status is PENDING.
  Testable: yes - property

5.2 WHEN a Worker_Service consumes a job, THE task status SHALL transition to RUNNING
  Thoughts: This is a rule that should apply to all consumed jobs. We can verify the status transition.
  Testable: yes - property

5.3 WHEN a Worker_Service completes a job, THE task status SHALL transition to COMPLETED
  Thoughts: This is a rule that should apply to all completed jobs. We can verify the status transition.
  Testable: yes - property

6.1 IF a PENDING task has no corresponding job in Redis, THEN THE Reconciler SHALL enqueue the job
  Thoughts: This is a rule that should apply to all orphaned PENDING tasks. We can create a PENDING task without a job, run reconciler, then verify the job is enqueued.
  Testable: yes - property

6.2 IF a RUNNING task has no corresponding job in Redis, THEN THE Reconciler SHALL reset status to PENDING and re-enqueue
  Thoughts: This is a rule that should apply to all orphaned RUNNING tasks. We can create a RUNNING task without a job, run reconciler, then verify it's reset to PENDING.
  Testable: yes - property

7.1 WHEN an inference endpoint receives a request, THE ML_Service SHALL acquire the GPU semaphore before loading models
  Thoughts: This is a rule that should apply to all inference requests. We can verify that concurrent requests are limited by the semaphore.
  Testable: yes - property

8.1 WHEN the Worker_Service receives inference results from ML_Service, THE Worker_Service SHALL extract config_hash and input_hash
  Thoughts: This is a rule that should apply to all inference results. We can verify that artifacts contain these hashes.
  Testable: yes - property

8.2 WHEN artifacts are inserted, THE artifacts table transaction SHALL be atomic (all-or-nothing)
  Thoughts: This is a rule that should apply to all artifact insertions. We can verify that either all artifacts are inserted or none.
  Testable: yes - property

9.1 WHEN the API_Service starts, THE API_Service SHALL NOT start arq consumer
  Thoughts: This is a rule about service initialization. We can verify that the API Service process does not have arq consumer running.
  Testable: yes - example

10.1 WHEN an operator calls POST /tasks/{task_id}/cancel, THE API_Service SHALL mark task as CANCELLED in PostgreSQL
  Thoughts: This is a rule that should apply to all cancellation requests. We can verify the task status is CANCELLED.
  Testable: yes - property

10.2 IF a job is queued, THEN Job.abort() SHALL prevent it from being picked up
  Thoughts: This is a rule that should apply to all queued jobs. We can abort a queued job and verify it's not processed.
  Testable: yes - property

11.1 WHEN services start, THE services SHALL emit structured logs with timestamps and log levels
  Thoughts: This is a rule about logging. We can verify that logs contain required fields.
  Testable: yes - property

12.1 THE API_Service SHALL read DATABASE_URL from environment for PostgreSQL connection
  Thoughts: This is a rule about configuration. We can verify that the service reads the environment variable.
  Testable: yes - example

13.1 WHEN the system is deployed, THE existing REST API endpoints SHALL remain unchanged
  Thoughts: This is a rule about backward compatibility. We can verify that existing endpoints still work.
  Testable: yes - property

14.1 WHEN the ML_Service is unavailable, THE Worker_Service SHALL retry the job with exponential backoff
  Thoughts: This is a rule that should apply to all failed requests. We can simulate ML Service unavailability and verify retries.
  Testable: yes - property

15.1 WHEN multiple Worker_Service instances are deployed, THE instances SHALL consume jobs from the same Redis queue
  Thoughts: This is a rule about scalability. We can verify that multiple workers consume from the same queue without duplicates.
  Testable: yes - property

16.1 WHEN a job is acknowledged in Redis, THE corresponding task SHALL be marked COMPLETED in PostgreSQL
  Thoughts: This is a round-trip property. We can verify that job completion is reflected in the database.
  Testable: yes - property

17.1 WHEN the ML_Service starts, THE ML_Service SHALL download all required models
  Thoughts: This is a rule that should apply to all ML Service startups. We can verify that all 6 models are downloaded.
  Testable: yes - property

17.2 WHEN the ML_Service downloads models, THE ML_Service SHALL cache them locally
  Thoughts: This is a rule about caching. We can verify that models are cached and not re-downloaded.
  Testable: yes - property

17.3 WHEN the ML_Service starts, THE ML_Service SHALL verify that each model loads successfully
  Thoughts: This is a rule that should apply to all startups. We can verify that all models load without errors.
  Testable: yes - property

17.4 WHEN the ML_Service starts, THE ML_Service SHALL detect GPU availability
  Thoughts: This is a rule about GPU detection. We can verify that GPU availability is correctly detected.
  Testable: yes - property

17.5 WHEN the ML_Service detects GPU availability, THE ML_Service SHALL log GPU device info
  Thoughts: This is a rule about logging. We can verify that GPU info is logged.
  Testable: yes - property

17.6 WHEN the ML_Service starts, THE ML_Service SHALL verify that each model can detect GPU or CPU
  Thoughts: This is a rule that should apply to all models. We can verify that models work on both GPU and CPU.
  Testable: yes - property

17.7 IF model download or verification fails, THE ML_Service SHALL fail startup
  Thoughts: This is a rule about error handling. We can verify that startup fails if models fail to initialize.
  Testable: yes - property

17.8 WHEN the ML_Service receives GET /health request, THE ML_Service SHALL return model status
  Thoughts: This is a rule about the health endpoint. We can verify that model status is returned.
  Testable: yes - property

17.9 WHEN the ML_Service receives GET /health request, THE ML_Service SHALL return GPU info
  Thoughts: This is a rule about the health endpoint. We can verify that GPU info is returned.
  Testable: yes - property

17.10 IF any model failed to initialize, THE GET /health endpoint SHALL return degraded status
  Thoughts: This is a rule about health status. We can verify that degraded status is returned when models fail.
  Testable: yes - property

18.1 WHEN the Worker_Service makes HTTP requests, THE Worker_Service SHALL use connection pooling
  Thoughts: This is a rule about resource management. We can verify that connection pooling is used.
  Testable: yes - example

19.1 WHEN a duplicate job is enqueued, THE arq queue SHALL recognize it as duplicate and not create a new job
  Thoughts: This is a rule about deduplication. We can enqueue the same job twice and verify only one exists.
  Testable: yes - property

20.1 WHEN the Worker_Service receives shutdown signal, THE Worker_Service SHALL finish current job before exiting
  Thoughts: This is a rule about graceful shutdown. We can send shutdown signal and verify the job completes.
  Testable: yes - property

#### Property Reflection

Reviewing all testable properties, I identify the following redundancies and consolidations:

- Properties 5.1, 5.2, 5.3 can be consolidated into a single "Task Status Lifecycle" property that verifies all transitions
- Properties 6.1, 6.2 can be consolidated into a single "Reconciliation Recovery" property
- Properties 8.1, 8.2 can be consolidated into a single "Artifact Persistence" property
- Properties 10.1, 10.2 can be consolidated into a single "Task Cancellation" property
- Properties 14.1 is already comprehensive for error handling

After consolidation, we have 16 core properties that provide comprehensive coverage without redundancy.

### Correctness Properties

**Property 1: Video Discovery Auto-Creates Tasks**

*For any* video path, when the discovery service processes it, the system SHALL create exactly 6 task records (one for each ML operation type) with status=PENDING and enqueue 6 corresponding jobs to Redis.

**Validates: Requirements 1.2, 1.3**

**Property 2: Job Enqueueing with Deduplication**

*For any* task, when enqueued to Redis, the system SHALL assign a unique job_id in format "ml_{task_id}" and prevent duplicate jobs from being created if the same task is enqueued twice.

**Validates: Requirements 1.3, 4.1, 19.1**

**Property 3: Task Status Lifecycle**

*For any* task, the status transitions SHALL follow the valid lifecycle: PENDING → RUNNING → COMPLETED, with optional transitions to FAILED or CANCELLED, and timestamps SHALL be recorded at each transition.

**Validates: Requirements 5.1, 5.2, 5.3, 5.7, 5.8**

**Property 4: Worker Job Consumption and Acknowledgment**

*For any* job in Redis, when consumed by a Worker Service, the corresponding task status SHALL transition to RUNNING, and upon completion, the job SHALL be acknowledged (XACK) and removed from the pending list.

**Validates: Requirements 2.1, 2.6, 5.2, 5.3**

**Property 5: ML Service Response Provenance**

*For any* inference request to the ML Service, the response SHALL include run_id, config_hash, input_hash, model_profile, producer, and producer_version fields, enabling reproducibility and traceability.

**Validates: Requirements 3.8, 3.9, 8.1**

**Property 6: Artifact Transformation and Persistence**

*For any* ML Service response, the Worker Service SHALL transform it into ArtifactEnvelopes, batch insert them atomically to PostgreSQL, and update the task status to COMPLETED in a single transaction.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 16.1**

**Property 7: Reconciliation Recovery**

*For any* PENDING task without a corresponding job in Redis, the Reconciler SHALL re-enqueue it. *For any* RUNNING task without a corresponding job in Redis, the Reconciler SHALL reset it to PENDING and re-enqueue it.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6**

**Property 8: GPU Semaphore Concurrency Control**

*For any* number of concurrent inference requests to the ML Service, the system SHALL limit simultaneous GPU operations to the configured semaphore value (default=2), queuing excess requests until slots become available.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

**Property 9: Task Cancellation**

*For any* task in PENDING or RUNNING status, when cancelled via API, the system SHALL mark it as CANCELLED in PostgreSQL and abort the corresponding job in Redis (preventing pickup if queued, or raising CancelledError if running).

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

**Property 10: Error Handling and Retry**

*For any* failed job (ML Service unavailable, network error, etc.), the Worker Service SHALL allow arq to retry with exponential backoff up to max_tries, and mark the task as FAILED only after retries are exhausted.

**Validates: Requirements 14.1, 14.2, 14.3, 14.7**

**Property 11: Horizontal Scaling**

*For any* number of Worker Service instances consuming from the same Redis queue, the system SHALL distribute jobs among instances without processing the same job twice (via arq consumer groups).

**Validates: Requirements 15.1, 15.2**

**Property 12: Backward Compatibility**

*For any* existing REST API endpoint (videos, tasks, artifacts, paths), the response format and behavior SHALL remain unchanged after service separation.

**Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.6**

**Property 13: Model Initialization on Boot**

*For any* ML Service instance, all required models SHALL be downloaded and verified during startup. GPU availability SHALL be detected and logged. If any model fails to initialize, the service SHALL fail startup rather than starting in a degraded state.

**Validates: Requirements 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7**

**Property 14: Job Idempotency**

*For any* task that is already COMPLETED, if the Worker Service processes it again (e.g., due to duplicate job), the system SHALL skip processing and return success without re-executing inference.

**Validates: Requirements 19.2, 19.3**

**Property 15: Graceful Shutdown**

*For any* Worker Service receiving a shutdown signal, the system SHALL complete the current job before exiting, not acknowledge incomplete jobs, and allow another worker to pick up the incomplete job from Redis.

**Validates: Requirements 20.1, 20.2, 20.3**

**Property 16: Service Isolation**

*For any* deployment, the API Service SHALL NOT run arq consumer, the Worker Service SHALL NOT expose HTTP endpoints, and the ML Service SHALL NOT access PostgreSQL directly.

**Validates: Requirements 9.3, 9.4, 9.10, 9.11**

## Error Handling

### Failure Scenarios and Recovery

| Scenario | Detection | Recovery | Result |
|----------|-----------|----------|--------|
| Redis data loss | Reconciler finds PENDING task without job | Re-enqueue from PostgreSQL | Job restarted |
| Worker crashes mid-job | Reconciler finds RUNNING task without job | Reset to PENDING, re-enqueue | Job restarted |
| ML Service unavailable | HTTP request fails | arq retries with backoff | Job retried up to max_tries |
| Job exceeds timeout | arq detects timeout | Mark task as FAILED | Operator can retry via API |
| Duplicate job enqueued | arq deduplication | Prevent duplicate | Single job processed |
| Task cancelled mid-execution | Job.abort() called | Raise CancelledError in worker | Task marked CANCELLED |
| Artifact batch insert fails | Transaction rollback | Mark task as FAILED | Operator can retry |

### Error Logging

All services SHALL emit structured logs with:
- Timestamp (ISO 8601)
- Log level (DEBUG, INFO, WARNING, ERROR)
- Service name
- Request/Job ID for traceability
- Error message and stack trace (for errors)

## Testing Strategy

### Unit Testing

Unit tests verify specific examples and edge cases:

1. **Task Creation**: Verify 6 tasks created per video with correct defaults
2. **Job Enqueueing**: Verify job_id format and deduplication
3. **Status Transitions**: Verify valid transitions and invalid ones are rejected
4. **Artifact Transformation**: Verify ML response → ArtifactEnvelope conversion
5. **Schema Validation**: Verify payload_json validation against schema models
6. **Reconciliation**: Verify orphaned task detection and re-enqueueing
7. **Error Handling**: Verify retry logic and failure marking
8. **Cancellation**: Verify task cancellation and job abort

### Property-Based Testing

Property tests verify universal properties across many generated inputs:

1. **Property 1**: Generate random video paths, verify 6 tasks created
2. **Property 2**: Generate random task IDs, verify deduplication
3. **Property 3**: Generate random task sequences, verify status lifecycle
4. **Property 4**: Generate random jobs, verify consumption and acknowledgment
5. **Property 5**: Generate random inference requests, verify response structure
6. **Property 6**: Generate random ML responses, verify transformation and persistence
7. **Property 7**: Generate random PENDING/RUNNING tasks, verify reconciliation
8. **Property 8**: Generate concurrent inference requests, verify semaphore limits
9. **Property 9**: Generate cancellation requests, verify task cancellation
10. **Property 10**: Generate failed requests, verify retry logic
11. **Property 11**: Generate multi-worker scenarios, verify no duplicates
12. **Property 12**: Generate API requests, verify backward compatibility
13. **Property 13**: Generate inference requests, verify lazy loading
14. **Property 14**: Generate duplicate jobs, verify idempotency
15. **Property 15**: Generate shutdown signals, verify graceful shutdown
16. **Property 16**: Generate service deployments, verify isolation

### Integration Testing

Integration tests verify end-to-end flows:

1. **Video Discovery → Task Creation → Job Enqueueing**: Full flow
2. **Job Consumption → ML Inference → Artifact Persistence**: Full flow
3. **Reconciliation Recovery**: Simulate Redis data loss and verify recovery
4. **Multi-Worker Scaling**: Deploy multiple workers and verify load distribution
5. **Task Cancellation**: Cancel task and verify job abort
6. **Error Recovery**: Simulate ML Service failure and verify retry

## Deployment & Operations

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: eioku
      POSTGRES_USER: eioku
      POSTGRES_PASSWORD: eioku_dev
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - eioku-network

  valkey:
    image: valkey/valkey:8-alpine
    command: valkey-server --appendonly yes
    volumes:
      - valkey_data:/data
    networks:
      - eioku-network

  api:
    build:
      context: ./backend
      dockerfile: ../dev/Dockerfile.api
    environment:
      DATABASE_URL: postgresql://eioku:eioku_dev@postgres:5432/eioku
      REDIS_URL: redis://valkey:6379/0
      ML_SERVICE_URL: http://ml-service:8001
    depends_on:
      - postgres
      - valkey
    ports:
      - "8000:8000"
    networks:
      - eioku-network

  worker-gpu:
    build:
      context: ./backend
      dockerfile: ../dev/Dockerfile.worker
    environment:
      DATABASE_URL: postgresql://eioku:eioku_dev@postgres:5432/eioku
      REDIS_URL: redis://valkey:6379/0
      ML_SERVICE_URL: http://ml-service:8001
      GPU_MODE: gpu
      ARQ_MAX_JOBS: 4
      ARQ_JOB_TIMEOUT: 1800
    depends_on:
      - postgres
      - valkey
      - ml-service
    networks:
      - eioku-network

  worker-cpu:
    build:
      context: ./backend
      dockerfile: ../dev/Dockerfile.worker
    environment:
      DATABASE_URL: postgresql://eioku:eioku_dev@postgres:5432/eioku
      REDIS_URL: redis://valkey:6379/0
      ML_SERVICE_URL: http://ml-service:8001
      GPU_MODE: cpu
      ARQ_MAX_JOBS: 8
      ARQ_JOB_TIMEOUT: 1800
    depends_on:
      - postgres
      - valkey
      - ml-service
    networks:
      - eioku-network

  ml-service:
    build:
      context: ./ml-service
    environment:
      GPU_CONCURRENCY: 2
      MODEL_CACHE_DIR: /models
    volumes:
      - ../test-videos:/media:ro
      - ml_models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8001:8001"
    networks:
      - eioku-network

  frontend:
    build:
      context: ./frontend
    ports:
      - "3000:3000"
    networks:
      - eioku-network

  nginx:
    image: nginx:alpine
    volumes:
      - ./dev/nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "8080:80"
    depends_on:
      - api
      - frontend
    networks:
      - eioku-network

networks:
  eioku-network:
    driver: bridge

volumes:
  postgres_data:
  valkey_data:
  ml_models:
```

### Environment Variables

**API Service:**
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis/Valkey connection string
- `ML_SERVICE_URL`: ML Service base URL
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

**Worker Service:**
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis/Valkey connection string
- `ML_SERVICE_URL`: ML Service base URL
- `GPU_MODE`: Job queue mode (gpu, cpu, or auto) - default: auto
- `ARQ_MAX_JOBS`: Max concurrent jobs (default=4)
- `ARQ_JOB_TIMEOUT`: Job timeout in seconds (default=1800)
- `LOG_LEVEL`: Logging level

**ML Service:**
- `GPU_CONCURRENCY`: Max concurrent GPU operations (default=2)
- `MODEL_CACHE_DIR`: Directory for model storage
- `REQUIRE_GPU`: Fail startup if GPU not available (default=false, set to true to require GPU)
- `LOG_LEVEL`: Logging level



## API Specifications

### ML Service OpenAPI Specification

The ML Service exposes a well-defined OpenAPI 3.0 specification for all inference endpoints. This enables:
- Auto-generated client libraries (Python, TypeScript, etc.)
- API documentation via Swagger UI
- Type-safe communication between Worker Service and ML Service
- Contract-first development

#### OpenAPI Schema File

```yaml
# ml-service/openapi.yaml
openapi: 3.0.0
info:
  title: Eioku ML Service API
  version: 1.0.0
  description: Stateless ML inference endpoints for video analysis

servers:
  - url: http://ml-service:8001
    description: Internal ML Service

paths:
  /health:
    get:
      summary: Health check endpoint
      operationId: getHealth
      tags:
        - health
      responses:
        '200':
          description: Service is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

  /infer/objects:
    post:
      summary: Detect objects in video
      operationId: detectObjects
      tags:
        - inference
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ObjectDetectionRequest'
      responses:
        '200':
          description: Object detection results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ObjectDetectionResponse'
        '400':
          description: Invalid request
        '500':
          description: Inference error

  /infer/faces:
    post:
      summary: Detect faces in video
      operationId: detectFaces
      tags:
        - inference
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/FaceDetectionRequest'
      responses:
        '200':
          description: Face detection results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FaceDetectionResponse'

  /infer/transcribe:
    post:
      summary: Transcribe audio from video
      operationId: transcribeVideo
      tags:
        - inference
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TranscriptionRequest'
      responses:
        '200':
          description: Transcription results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TranscriptionResponse'

  /infer/ocr:
    post:
      summary: Extract text from video frames
      operationId: extractOCR
      tags:
        - inference
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/OCRRequest'
      responses:
        '200':
          description: OCR results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/OCRResponse'

  /infer/places:
    post:
      summary: Classify places in video frames
      operationId: classifyPlaces
      tags:
        - inference
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PlaceDetectionRequest'
      responses:
        '200':
          description: Place classification results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PlaceDetectionResponse'

  /infer/scenes:
    post:
      summary: Detect scene boundaries in video
      operationId: detectScenes
      tags:
        - inference
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SceneDetectionRequest'
      responses:
        '200':
          description: Scene detection results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SceneDetectionResponse'

components:
  schemas:
    HealthResponse:
      type: object
      required:
        - status
        - models_loaded
        - gpu_available
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        models_loaded:
          type: array
          items:
            type: string
        gpu_available:
          type: boolean
        gpu_memory_used_mb:
          type: integer

    BoundingBox:
      type: object
      required:
        - x
        - y
        - width
        - height
      properties:
        x:
          type: number
          format: float
        y:
          type: number
          format: float
        width:
          type: number
          format: float
        height:
          type: number
          format: float

    Detection:
      type: object
      required:
        - frame_index
        - timestamp_ms
        - label
        - confidence
        - bbox
      properties:
        frame_index:
          type: integer
        timestamp_ms:
          type: integer
        label:
          type: string
        confidence:
          type: number
          format: float
          minimum: 0
          maximum: 1
        bbox:
          $ref: '#/components/schemas/BoundingBox'

    ObjectDetectionRequest:
      type: object
      required:
        - video_path
        - model_name
      properties:
        video_path:
          type: string
          description: Path to video file
        model_name:
          type: string
          description: Model file name (e.g., yolov8n.pt)
        frame_interval:
          type: integer
          default: 30
          description: Process every Nth frame
        confidence_threshold:
          type: number
          format: float
          default: 0.5
          minimum: 0
          maximum: 1
        model_profile:
          type: string
          enum: [fast, balanced, high_quality]
          default: balanced

    ObjectDetectionResponse:
      type: object
      required:
        - run_id
        - config_hash
        - input_hash
        - model_profile
        - producer
        - producer_version
        - detections
      properties:
        run_id:
          type: string
          format: uuid
        config_hash:
          type: string
          description: Hash of configuration for reproducibility
        input_hash:
          type: string
          description: Hash of input video for reproducibility
        model_profile:
          type: string
        producer:
          type: string
        producer_version:
          type: string
        detections:
          type: array
          items:
            $ref: '#/components/schemas/Detection'

    FaceDetectionRequest:
      type: object
      required:
        - video_path
        - model_name
      properties:
        video_path:
          type: string
        model_name:
          type: string
        frame_interval:
          type: integer
          default: 30
        confidence_threshold:
          type: number
          format: float
          default: 0.5

    FaceDetectionResponse:
      type: object
      required:
        - run_id
        - config_hash
        - input_hash
        - detections
      properties:
        run_id:
          type: string
          format: uuid
        config_hash:
          type: string
        input_hash:
          type: string
        detections:
          type: array
          items:
            allOf:
              - $ref: '#/components/schemas/Detection'
              - type: object
                properties:
                  cluster_id:
                    type: string

    Segment:
      type: object
      required:
        - start_ms
        - end_ms
        - text
        - confidence
      properties:
        start_ms:
          type: integer
        end_ms:
          type: integer
        text:
          type: string
        confidence:
          type: number
          format: float
        words:
          type: array
          items:
            type: object
            properties:
              word:
                type: string
              start_ms:
                type: integer
              end_ms:
                type: integer
              confidence:
                type: number
                format: float

    TranscriptionRequest:
      type: object
      required:
        - video_path
        - model_name
      properties:
        video_path:
          type: string
        model_name:
          type: string
          description: Whisper model size (tiny, base, small, medium, large)
        language:
          type: string
          nullable: true
          description: ISO 639-1 language code (null for auto-detect)
        vad_filter:
          type: boolean
          default: true

    TranscriptionResponse:
      type: object
      required:
        - run_id
        - config_hash
        - input_hash
        - language
        - segments
      properties:
        run_id:
          type: string
          format: uuid
        config_hash:
          type: string
        input_hash:
          type: string
        language:
          type: string
        segments:
          type: array
          items:
            $ref: '#/components/schemas/Segment'

    OCRRequest:
      type: object
      required:
        - video_path
      properties:
        video_path:
          type: string
        frame_interval:
          type: integer
          default: 60
        languages:
          type: array
          items:
            type: string
          default: [en]
        use_gpu:
          type: boolean
          default: true

    OCRResponse:
      type: object
      required:
        - run_id
        - config_hash
        - input_hash
        - detections
      properties:
        run_id:
          type: string
          format: uuid
        config_hash:
          type: string
        input_hash:
          type: string
        detections:
          type: array
          items:
            type: object
            properties:
              frame_index:
                type: integer
              timestamp_ms:
                type: integer
              text:
                type: string
              confidence:
                type: number
                format: float
              polygon:
                type: array
                items:
                  type: object
                  properties:
                    x:
                      type: number
                    y:
                      type: number

    PlaceDetectionRequest:
      type: object
      required:
        - video_path
      properties:
        video_path:
          type: string
        frame_interval:
          type: integer
          default: 60
        top_k:
          type: integer
          default: 5

    PlaceDetectionResponse:
      type: object
      required:
        - run_id
        - config_hash
        - input_hash
        - classifications
      properties:
        run_id:
          type: string
          format: uuid
        config_hash:
          type: string
        input_hash:
          type: string
        classifications:
          type: array
          items:
            type: object
            properties:
              frame_index:
                type: integer
              timestamp_ms:
                type: integer
              predictions:
                type: array
                items:
                  type: object
                  properties:
                    label:
                      type: string
                    confidence:
                      type: number
                      format: float

    Scene:
      type: object
      required:
        - scene_index
        - start_ms
        - end_ms
      properties:
        scene_index:
          type: integer
        start_ms:
          type: integer
        end_ms:
          type: integer

    SceneDetectionRequest:
      type: object
      required:
        - video_path
      properties:
        video_path:
          type: string
        threshold:
          type: number
          format: float
          default: 0.4
        min_scene_length:
          type: number
          format: float
          default: 0.6

    SceneDetectionResponse:
      type: object
      required:
        - run_id
        - config_hash
        - input_hash
        - scenes
      properties:
        run_id:
          type: string
          format: uuid
        config_hash:
          type: string
        input_hash:
          type: string
        scenes:
          type: array
          items:
            $ref: '#/components/schemas/Scene'
```

### Generated ML Service Client

The OpenAPI spec is used to generate a type-safe Python client:

```python
# ml-service/generated/client.py (auto-generated from openapi.yaml)
# Generated using: openapi-python-client generate --path openapi.yaml

from ml_service.generated.models import (
    ObjectDetectionRequest,
    ObjectDetectionResponse,
    FaceDetectionRequest,
    FaceDetectionResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    OCRRequest,
    OCRResponse,
    PlaceDetectionRequest,
    PlaceDetectionResponse,
    SceneDetectionRequest,
    SceneDetectionResponse,
    HealthResponse
)

class MLServiceClient:
    """Auto-generated client for ML Service API"""
    
    async def get_health(self) -> HealthResponse:
        """Get health status"""
        ...
    
    async def detect_objects(self, request: ObjectDetectionRequest) -> ObjectDetectionResponse:
        """Detect objects in video"""
        ...
    
    async def detect_faces(self, request: FaceDetectionRequest) -> FaceDetectionResponse:
        """Detect faces in video"""
        ...
    
    async def transcribe_video(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Transcribe audio from video"""
        ...
    
    async def extract_ocr(self, request: OCRRequest) -> OCRResponse:
        """Extract text from video frames"""
        ...
    
    async def classify_places(self, request: PlaceDetectionRequest) -> PlaceDetectionResponse:
        """Classify places in video frames"""
        ...
    
    async def detect_scenes(self, request: SceneDetectionRequest) -> SceneDetectionResponse:
        """Detect scene boundaries in video"""
        ...
```

### Worker Service ML Client Usage

The Worker Service uses the generated client for type-safe communication:

```python
# backend/src/services/ml_client.py
from ml_service.generated.client import MLServiceClient
from ml_service.generated.models import (
    ObjectDetectionRequest,
    FaceDetectionRequest,
    TranscriptionRequest,
    OCRRequest,
    PlaceDetectionRequest,
    SceneDetectionRequest
)

class WorkerMLClient:
    """Wrapper around generated ML Service client"""
    
    def __init__(self, base_url: str):
        self.client = MLServiceClient(base_url=base_url)
    
    async def infer_objects(self, video_path: str, config: dict):
        """Dispatch object detection task"""
        request = ObjectDetectionRequest(
            video_path=video_path,
            model_name=config.get('model_name', 'yolov8n.pt'),
            frame_interval=config.get('frame_interval', 30),
            confidence_threshold=config.get('confidence_threshold', 0.5),
            model_profile=config.get('model_profile', 'balanced')
        )
        return await self.client.detect_objects(request)
    
    async def infer_faces(self, video_path: str, config: dict):
        """Dispatch face detection task"""
        request = FaceDetectionRequest(
            video_path=video_path,
            model_name=config.get('model_name', 'yolov8n-face.pt'),
            frame_interval=config.get('frame_interval', 30),
            confidence_threshold=config.get('confidence_threshold', 0.5)
        )
        return await self.client.detect_faces(request)
    
    async def infer_transcription(self, video_path: str, config: dict):
        """Dispatch transcription task"""
        request = TranscriptionRequest(
            video_path=video_path,
            model_name=config.get('model_name', 'large-v3'),
            language=config.get('language'),
            vad_filter=config.get('vad_filter', True)
        )
        return await self.client.transcribe_video(request)
    
    async def infer_ocr(self, video_path: str, config: dict):
        """Dispatch OCR task"""
        request = OCRRequest(
            video_path=video_path,
            frame_interval=config.get('frame_interval', 60),
            languages=config.get('languages', ['en']),
            use_gpu=config.get('use_gpu', True)
        )
        return await self.client.extract_ocr(request)
    
    async def infer_places(self, video_path: str, config: dict):
        """Dispatch place detection task"""
        request = PlaceDetectionRequest(
            video_path=video_path,
            frame_interval=config.get('frame_interval', 60),
            top_k=config.get('top_k', 5)
        )
        return await self.client.classify_places(request)
    
    async def infer_scenes(self, video_path: str, config: dict):
        """Dispatch scene detection task"""
        request = SceneDetectionRequest(
            video_path=video_path,
            threshold=config.get('threshold', 0.4),
            min_scene_length=config.get('min_scene_length', 0.6)
        )
        return await self.client.detect_scenes(request)
    
    async def health_check(self):
        """Check ML Service health"""
        return await self.client.get_health()
```

### Client Generation Build Step

The OpenAPI client is generated as part of the build process:

```bash
# ml-service/Dockerfile
FROM python:3.11-slim

# Install openapi-python-client for code generation
RUN pip install openapi-python-client

# Copy OpenAPI spec
COPY openapi.yaml /app/openapi.yaml

# Generate client code
RUN openapi-python-client generate --path /app/openapi.yaml --output-dir /app/generated

# Rest of Dockerfile...
```

### Benefits of OpenAPI + Generated Clients

1. **Type Safety**: Generated models ensure type correctness at compile time
2. **Contract-First**: API contract is the source of truth
3. **Documentation**: Swagger UI auto-generated from spec
4. **Versioning**: Easy to version API and maintain backward compatibility
5. **Consistency**: All clients use same generated code
6. **Validation**: Pydantic models validate requests/responses
7. **IDE Support**: Full autocomplete and type hints in IDEs

