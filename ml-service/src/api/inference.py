"""Inference endpoints for ML Service."""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from ..models.requests import (
    FaceDetectionRequest,
    ObjectDetectionRequest,
    OCRRequest,
    PlaceDetectionRequest,
    SceneDetectionRequest,
    TranscriptionRequest,
)
from ..models.responses import (
    Detection,
    FaceDetection,
    FaceDetectionResponse,
    ObjectDetectionResponse,
    OCRDetection,
    OCRResponse,
    PlaceClassification,
    PlacePrediction,
    PlaceDetectionResponse,
    Scene,
    SceneDetectionResponse,
    Segment,
    TranscriptionResponse,
)
from ..utils.hashing import compute_config_hash, compute_input_hash

if TYPE_CHECKING:
    from ..services.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/infer", tags=["inference"])

# Global references (set by main.py)
GPU_SEMAPHORE: asyncio.Semaphore | None = None
MODEL_MANAGER: "ModelManager | None" = None
GPU_AVAILABLE: bool = False


def set_globals(semaphore: asyncio.Semaphore, manager: "ModelManager", gpu_available: bool = False):
    """Set global references for inference endpoints.

    Args:
        semaphore: GPU semaphore for concurrency control
        manager: ModelManager instance
        gpu_available: Whether GPU is available for inference
    """
    global GPU_SEMAPHORE, MODEL_MANAGER, GPU_AVAILABLE
    GPU_SEMAPHORE = semaphore
    MODEL_MANAGER = manager
    GPU_AVAILABLE = gpu_available


def _get_device() -> str:
    """Get device string for model inference.
    
    Returns "cuda" if GPU is available, "cpu" otherwise.
    This ensures models respect our GPU availability flag.
    """
    return "cuda" if GPU_AVAILABLE else "cpu"


async def _acquire_gpu_if_available():
    """Context manager that acquires GPU semaphore only if GPU is available.
    
    Returns a context manager that either acquires the GPU semaphore (if GPU available)
    or does nothing (if CPU-only mode).
    """
    class GPUContextManager:
        def __init__(self, semaphore: asyncio.Semaphore | None, gpu_available: bool):
            self.semaphore = semaphore
            self.gpu_available = gpu_available
            self.acquired = False
        
        async def __aenter__(self):
            if self.gpu_available and self.semaphore is not None:
                await self.semaphore.acquire()
                self.acquired = True
                logger.debug("GPU semaphore acquired")
            else:
                logger.debug("Running on CPU (GPU not available or semaphore not initialized)")
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.acquired and self.semaphore is not None:
                self.semaphore.release()
                logger.debug("GPU semaphore released")
            return False
    
    return GPUContextManager(GPU_SEMAPHORE, GPU_AVAILABLE)


@router.post("/objects", response_model=ObjectDetectionResponse)
async def detect_objects(request: ObjectDetectionRequest) -> ObjectDetectionResponse:
    """Detect objects in video using YOLO.

    Args:
        request: Object detection request

    Returns:
        ObjectDetectionResponse with detections and provenance metadata
    """
    try:
        async with await _acquire_gpu_if_available():
            from ultralytics import YOLO

            device = _get_device()
            logger.info(f"Object detection: {request.video_path} (device: {device})")

            # Load model with explicit device
            model = YOLO(request.model_name)
            model.to(device)

            # Run inference with streaming to avoid memory issues
            results = model(
                request.video_path,
                conf=request.confidence_threshold,
                verbose=False,
                device=device,
                stream=True,
            )

            # Extract detections
            detections = []
            for frame_idx, result in enumerate(results):
                timestamp_ms = int((frame_idx / 30) * 1000)  # Approximate timestamp

                for box in result.boxes:
                    detection = Detection(
                        frame_index=frame_idx,
                        timestamp_ms=timestamp_ms,
                        label=result.names[int(box.cls)],
                        confidence=float(box.conf),
                        bbox={
                            "x": float(box.xyxy[0][0]),
                            "y": float(box.xyxy[0][1]),
                            "width": float(box.xyxy[0][2] - box.xyxy[0][0]),
                            "height": float(box.xyxy[0][3] - box.xyxy[0][1]),
                        },
                    )
                    detections.append(detection)

            # Compute hashes
            config = {
                "model_name": request.model_name,
                "frame_interval": request.frame_interval,
                "confidence_threshold": request.confidence_threshold,
                "model_profile": request.model_profile,
            }
            config_hash = compute_config_hash(config)
            input_hash = compute_input_hash(request.video_path)

            return ObjectDetectionResponse(
                run_id=str(uuid.uuid4()),
                config_hash=config_hash,
                input_hash=input_hash,
                model_profile=request.model_profile,
                producer="yolo",
                producer_version="8.0.0",
                detections=detections,
            )

    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        raise HTTPException(500, f"Object detection failed: {str(e)}")


@router.post("/faces", response_model=FaceDetectionResponse)
async def detect_faces(request: FaceDetectionRequest) -> FaceDetectionResponse:
    """Detect faces in video using YOLO.

    Args:
        request: Face detection request

    Returns:
        FaceDetectionResponse with detections and provenance metadata
    """
    try:
        async with await _acquire_gpu_if_available():
            from ultralytics import YOLO

            device = _get_device()
            logger.info(f"Face detection: {request.video_path} (device: {device})")

            # Load model with explicit device
            model = YOLO(request.model_name)
            model.to(device)

            # Run inference with streaming to avoid memory issues
            results = model(
                request.video_path,
                conf=request.confidence_threshold,
                verbose=False,
                device=device,
                stream=True,
            )

            # Extract detections
            detections = []
            for frame_idx, result in enumerate(results):
                timestamp_ms = int((frame_idx / 30) * 1000)

                for box in result.boxes:
                    detection = FaceDetection(
                        frame_index=frame_idx,
                        timestamp_ms=timestamp_ms,
                        label="face",
                        confidence=float(box.conf),
                        bbox={
                            "x": float(box.xyxy[0][0]),
                            "y": float(box.xyxy[0][1]),
                            "width": float(box.xyxy[0][2] - box.xyxy[0][0]),
                            "height": float(box.xyxy[0][3] - box.xyxy[0][1]),
                        },
                        cluster_id=None,
                    )
                    detections.append(detection)

            # Compute hashes
            config = {
                "model_name": request.model_name,
                "frame_interval": request.frame_interval,
                "confidence_threshold": request.confidence_threshold,
            }
            config_hash = compute_config_hash(config)
            input_hash = compute_input_hash(request.video_path)

            return FaceDetectionResponse(
                run_id=str(uuid.uuid4()),
                config_hash=config_hash,
                input_hash=input_hash,
                producer="yolo",
                producer_version="8.0.0",
                detections=detections,
            )

    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(500, f"Face detection failed: {str(e)}")


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(request: TranscriptionRequest) -> TranscriptionResponse:
    """Transcribe audio from video using Whisper.

    Args:
        request: Transcription request

    Returns:
        TranscriptionResponse with segments and provenance metadata
    """
    try:
        async with await _acquire_gpu_if_available():
            from faster_whisper import WhisperModel

            device = _get_device()
            logger.info(f"Transcription: {request.video_path} (device: {device})")

            # Load model with explicit device
            model = WhisperModel(
                request.model_name, device=device, compute_type="auto"
            )

            # Run inference
            segments, info = model.transcribe(
                request.video_path,
                language=request.language,
                vad_filter=request.vad_filter,
            )

            # Extract segments
            transcription_segments = []
            for segment in segments:
                ts = Segment(
                    start_ms=int(segment.start * 1000),
                    end_ms=int(segment.end * 1000),
                    text=segment.text,
                    confidence=None,  # faster_whisper doesn't provide segment confidence
                    words=None,
                )
                transcription_segments.append(ts)

            # Compute hashes
            config = {
                "model_name": request.model_name,
                "language": request.language,
                "vad_filter": request.vad_filter,
            }
            config_hash = compute_config_hash(config)
            input_hash = compute_input_hash(request.video_path)

            return TranscriptionResponse(
                run_id=str(uuid.uuid4()),
                config_hash=config_hash,
                input_hash=input_hash,
                language=info.language,
                producer="whisper",
                producer_version="3.0",
                segments=transcription_segments,
            )

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@router.post("/ocr", response_model=OCRResponse)
async def extract_ocr(request: OCRRequest) -> OCRResponse:
    """Extract text from video frames using EasyOCR.

    Args:
        request: OCR request

    Returns:
        OCRResponse with detections and provenance metadata
    """
    try:
        async with await _acquire_gpu_if_available():
            import cv2
            import easyocr

            logger.info(f"OCR: {request.video_path} (GPU: {GPU_AVAILABLE})")

            # Load model with explicit GPU flag
            reader = easyocr.Reader(
                request.languages, gpu=GPU_AVAILABLE, verbose=False
            )

            # Open video
            cap = cv2.VideoCapture(request.video_path)
            frame_count = 0
            detections = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every Nth frame
                if frame_count % request.frame_interval == 0:
                    results = reader.readtext(frame)

                    timestamp_ms = int(
                        (frame_count / cap.get(cv2.CAP_PROP_FPS)) * 1000
                    )

                    for result in results:
                        bbox, text, confidence = result
                        detection = OCRDetection(
                            frame_index=frame_count,
                            timestamp_ms=timestamp_ms,
                            text=text,
                            confidence=confidence,
                            polygon=[{"x": float(p[0]), "y": float(p[1])} for p in bbox],
                        )
                        detections.append(detection)

                frame_count += 1

            cap.release()

            # Compute hashes
            config = {
                "frame_interval": request.frame_interval,
                "languages": request.languages,
                "use_gpu": request.use_gpu,
            }
            config_hash = compute_config_hash(config)
            input_hash = compute_input_hash(request.video_path)

            return OCRResponse(
                run_id=str(uuid.uuid4()),
                config_hash=config_hash,
                input_hash=input_hash,
                producer="easyocr",
                producer_version="1.7.0",
                detections=detections,
            )

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(500, f"OCR failed: {str(e)}")


@router.post("/places", response_model=PlaceDetectionResponse)
async def classify_places(request: PlaceDetectionRequest) -> PlaceDetectionResponse:
    """Classify places in video frames using Places365.

    Args:
        request: Place detection request

    Returns:
        PlaceDetectionResponse with classifications and provenance metadata
    """
    try:
        async with await _acquire_gpu_if_available():
            device = _get_device()
            logger.info(f"Place detection: {request.video_path} (device: {device})")

            # Placeholder implementation
            classifications = []

            # Compute hashes
            config = {
                "frame_interval": request.frame_interval,
                "top_k": request.top_k,
            }
            config_hash = compute_config_hash(config)
            input_hash = compute_input_hash(request.video_path)

            return PlaceDetectionResponse(
                run_id=str(uuid.uuid4()),
                config_hash=config_hash,
                input_hash=input_hash,
                producer="places365",
                producer_version="1.0.0",
                classifications=classifications,
            )

    except Exception as e:
        logger.error(f"Place detection failed: {e}")
        raise HTTPException(500, f"Place detection failed: {str(e)}")


@router.post("/scenes", response_model=SceneDetectionResponse)
async def detect_scenes(request: SceneDetectionRequest) -> SceneDetectionResponse:
    """Detect scene boundaries in video.

    Args:
        request: Scene detection request

    Returns:
        SceneDetectionResponse with scenes and provenance metadata
    """
    try:
        async with await _acquire_gpu_if_available():
            device = _get_device()
            logger.info(f"Scene detection: {request.video_path} (device: {device})")

            # Placeholder implementation
            scenes = []

            # Compute hashes
            config = {
                "threshold": request.threshold,
                "min_scene_length": request.min_scene_length,
            }
            config_hash = compute_config_hash(config)
            input_hash = compute_input_hash(request.video_path)

            return SceneDetectionResponse(
                run_id=str(uuid.uuid4()),
                config_hash=config_hash,
                input_hash=input_hash,
                producer="scenedetect",
                producer_version="0.6.0",
                scenes=scenes,
            )

    except Exception as e:
        logger.error(f"Scene detection failed: {e}")
        raise HTTPException(500, f"Scene detection failed: {str(e)}")
