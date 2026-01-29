"""Response models for ML Service inference endpoints."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    width: float = Field(..., description="Width")
    height: float = Field(..., description="Height")


class Detection(BaseModel):
    """Detection result with bounding box."""

    frame_index: int = Field(..., description="Frame index")
    timestamp_ms: int = Field(..., description="Timestamp in milliseconds")
    label: str = Field(..., description="Detection label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box")


class FaceDetection(Detection):
    """Face detection result with cluster ID."""

    cluster_id: str | None = Field(default=None, description="Face cluster ID")


class Segment(BaseModel):
    """Transcription segment."""

    start_ms: int = Field(..., description="Start time in milliseconds")
    end_ms: int = Field(..., description="End time in milliseconds")
    text: str = Field(..., description="Segment text")
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Confidence score (if available)"
    )
    words: list[dict] | None = Field(default=None, description="Word-level details")


class OCRDetection(BaseModel):
    """OCR detection result."""

    frame_index: int = Field(..., description="Frame index")
    timestamp_ms: int = Field(..., description="Timestamp in milliseconds")
    text: str = Field(..., description="Detected text")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    polygon: list[dict] = Field(..., description="Text polygon coordinates")


class PlacePrediction(BaseModel):
    """Place classification prediction."""

    label: str = Field(..., description="Place label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class PlaceClassification(BaseModel):
    """Place classification result."""

    frame_index: int = Field(..., description="Frame index")
    timestamp_ms: int = Field(..., description="Timestamp in milliseconds")
    predictions: list[PlacePrediction] = Field(..., description="Top predictions")


class Scene(BaseModel):
    """Scene detection result."""

    scene_index: int = Field(..., description="Scene index")
    start_ms: int = Field(..., description="Start time in milliseconds")
    end_ms: int = Field(..., description="End time in milliseconds")


class ObjectDetectionResponse(BaseModel):
    """Response model for object detection."""

    run_id: str = Field(..., description="Unique run ID")
    config_hash: str = Field(..., description="Configuration hash")
    input_hash: str = Field(..., description="Input hash")
    model_profile: str = Field(..., description="Model profile used")
    producer: str = Field(..., description="Producer name")
    producer_version: str = Field(..., description="Producer version")
    detections: list[Detection] = Field(..., description="List of detections")


class FaceDetectionResponse(BaseModel):
    """Response model for face detection."""

    run_id: str = Field(..., description="Unique run ID")
    config_hash: str = Field(..., description="Configuration hash")
    input_hash: str = Field(..., description="Input hash")
    producer: str = Field(default="yolo", description="Producer name")
    producer_version: str = Field(default="8.0.0", description="Producer version")
    detections: list[FaceDetection] = Field(..., description="List of detections")


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""

    run_id: str = Field(..., description="Unique run ID")
    config_hash: str = Field(..., description="Configuration hash")
    input_hash: str = Field(..., description="Input hash")
    language: str = Field(..., description="Detected language")
    producer: str = Field(default="whisper", description="Producer name")
    producer_version: str = Field(default="3.0", description="Producer version")
    segments: list[Segment] = Field(..., description="List of segments")


class OCRResponse(BaseModel):
    """Response model for OCR."""

    run_id: str = Field(..., description="Unique run ID")
    config_hash: str = Field(..., description="Configuration hash")
    input_hash: str = Field(..., description="Input hash")
    producer: str = Field(default="easyocr", description="Producer name")
    producer_version: str = Field(default="1.7.0", description="Producer version")
    detections: list[OCRDetection] = Field(..., description="List of detections")


class PlaceDetectionResponse(BaseModel):
    """Response model for place detection."""

    run_id: str = Field(..., description="Unique run ID")
    config_hash: str = Field(..., description="Configuration hash")
    input_hash: str = Field(..., description="Input hash")
    producer: str = Field(default="places365", description="Producer name")
    producer_version: str = Field(default="1.0.0", description="Producer version")
    classifications: list[PlaceClassification] = Field(
        ..., description="List of classifications"
    )


class SceneDetectionResponse(BaseModel):
    """Response model for scene detection."""

    run_id: str = Field(..., description="Unique run ID")
    config_hash: str = Field(..., description="Configuration hash")
    input_hash: str = Field(..., description="Input hash")
    producer: str = Field(default="scenedetect", description="Producer name")
    producer_version: str = Field(default="0.6.0", description="Producer version")
    scenes: list[Scene] = Field(..., description="List of scenes")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        ..., description="Service status (healthy, degraded, unhealthy)"
    )
    models_loaded: list[str] = Field(..., description="List of loaded models")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_device_name: str | None = Field(default=None, description="GPU device name")
    gpu_memory_total_mb: int | None = Field(
        default=None, description="Total GPU memory"
    )
    gpu_memory_used_mb: int | None = Field(default=None, description="Used GPU memory")


class AcceptedResponse(BaseModel):
    """Response for async job submission (202 Accepted)."""

    task_id: str = Field(..., description="Task identifier")
    status: str = Field(default="accepted", description="Job status")


class TaskInfo(BaseModel):
    """Information about a running ML task."""

    task_id: str = Field(..., description="Task identifier")
    endpoint: str = Field(..., description="Inference endpoint")
    pid: int = Field(..., description="Process ID")
    start_time: str = Field(..., description="Task start time (ISO format)")
    elapsed_seconds: float = Field(..., description="Elapsed time in seconds")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_mb: float = Field(..., description="Memory usage in MB")
    num_threads: int = Field(..., description="Number of threads")
    gpu_percent: float = Field(..., description="GPU usage percentage")
    gpu_memory_mb: float = Field(..., description="GPU memory usage in MB")


class TaskSummaryResponse(BaseModel):
    """Summary of all running ML tasks."""

    running_tasks: int = Field(..., description="Number of running tasks")
    total_cpu_percent: float = Field(..., description="Total CPU usage percentage")
    total_memory_mb: float = Field(..., description="Total memory usage in MB")
    total_gpu_percent: float = Field(..., description="Total GPU usage percentage")
    total_gpu_memory_mb: float = Field(..., description="Total GPU memory usage in MB")
    total_threads: int = Field(..., description="Total number of threads")
    tasks: list[TaskInfo] = Field(
        default_factory=list, description="List of running tasks"
    )
