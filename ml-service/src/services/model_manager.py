"""Model manager for downloading and verifying ML models."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model download, verification, and lifecycle."""

    def __init__(self, cache_dir: str = "/models"):
        """Initialize model manager.

        Args:
            cache_dir: Directory for caching downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self._gpu_available = None  # Lazy initialization

    @property
    def gpu_available(self) -> bool:
        """Check GPU availability (lazy initialization)."""
        if self._gpu_available is None:
            try:
                import torch

                self._gpu_available = torch.cuda.is_available()
            except Exception as e:
                logger.warning(f"Could not check GPU availability: {e}")
                self._gpu_available = False
        return self._gpu_available

    def _get_device(self) -> str:
        """Get device string for model loading.

        Returns:
            "cuda" if GPU available, "cpu" otherwise
        """
        return "cuda" if self.gpu_available else "cpu"

    async def download_model(self, model_name: str, model_type: str) -> Path:
        """Download model from source and cache locally.

        Args:
            model_name: Name of the model to download
            model_type: Type of model (yolo, whisper, places365, easyocr)

        Returns:
            Path to cached model
        """
        logger.info(f"Downloading {model_type} model: {model_name}")

        try:
            if model_type == "yolo":
                import os

                from ultralytics import YOLO

                # Set YOLO_HOME to cache directory
                yolo_cache = self.cache_dir / "ultralytics"
                yolo_cache.mkdir(parents=True, exist_ok=True)
                os.environ["YOLO_HOME"] = str(yolo_cache)

                # Download model - YOLO respects YOLO_HOME environment variable
                model = YOLO(model_name)

                # Get the actual model path from YOLO
                model_path = Path(model.model_name)
                if not model_path.is_absolute():
                    model_path = yolo_cache / model_name

                logger.info(f"✓ YOLO model {model_name} downloaded to {yolo_cache}")
                return model_path

            elif model_type == "whisper":
                # faster-whisper will use HF_HOME environment variable set in main.py
                from faster_whisper import WhisperModel

                WhisperModel(model_name, device=self._get_device(), compute_type="auto")
                logger.info(f"✓ Whisper model {model_name} downloaded")
                return Path(model_name)

            elif model_type == "easyocr":
                # EasyOCR will use EASYOCR_HOME environment variable set in main.py
                import easyocr

                easyocr.Reader(["en"], gpu=self.gpu_available, verbose=False)
                logger.info("✓ EasyOCR model downloaded")
                return Path("easyocr")

            elif model_type == "places365":
                # Places365 model is loaded via torchvision
                import torchvision.models as models

                models.resnet18(pretrained=False)
                logger.info(f"✓ Places365 model {model_name} downloaded")
                return Path(model_name)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise

    async def verify_model(self, model_name: str, model_type: str) -> bool:
        """Verify model loads and GPU detection works.

        Args:
            model_name: Name of the model to verify
            model_type: Type of model

        Returns:
            True if model verified successfully
        """
        logger.info(f"Verifying {model_type} model: {model_name}")

        try:
            if model_type == "yolo":
                from ultralytics import YOLO

                model = YOLO(model_name)
                # Test on dummy image
                import numpy as np

                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                model(dummy_image, verbose=False)
                logger.info(f"✓ YOLO model {model_name} verified")

            elif model_type == "whisper":
                from faster_whisper import WhisperModel

                WhisperModel(model_name, device=self._get_device(), compute_type="auto")
                logger.info(f"✓ Whisper model {model_name} verified")

            elif model_type == "easyocr":
                import easyocr

                easyocr.Reader(["en"], gpu=self.gpu_available, verbose=False)
                logger.info("✓ EasyOCR model verified")

            elif model_type == "places365":
                import torchvision.models as models

                model = models.resnet18(pretrained=False)
                model.to(self._get_device())
                model.eval()
                logger.info("✓ Places365 model verified")

            # Log GPU detection result
            if self.gpu_available:
                import torch

                device_name = torch.cuda.get_device_name(0)
                logger.info(f"  GPU detected: {device_name}")
            else:
                logger.info("  GPU not available, using CPU")

            return True

        except Exception as e:
            logger.error(f"✗ Model verification failed for {model_name}: {e}")
            raise

    def get_gpu_info(self) -> dict:
        """Get GPU information.

        Returns:
            Dictionary with GPU info
        """
        if not self.gpu_available:
            return {
                "gpu_available": False,
                "gpu_device_name": None,
                "gpu_memory_total_mb": None,
                "gpu_memory_used_mb": None,
            }

        import torch

        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e6
        allocated_memory = torch.cuda.memory_allocated(0) / 1e6

        return {
            "gpu_available": True,
            "gpu_device_name": device_name,
            "gpu_memory_total_mb": int(total_memory),
            "gpu_memory_used_mb": int(allocated_memory),
        }

    def detect_gpu(self) -> bool:
        """Detect GPU availability.

        Returns:
            True if GPU available, False otherwise
        """
        return self.gpu_available

    def log_gpu_info(self):
        """Log GPU information."""
        if self.gpu_available:
            import torch

            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU device: {device_name}")
            logger.info(f"GPU memory: {total_memory:.2f} GB")
        else:
            logger.warning("GPU not available - will use CPU for inference (slower)")

    async def detect_objects(self, video_path: str, config: dict) -> dict:
        """Detect objects in video using YOLO.

        Args:
            video_path: Path to video file
            config: Configuration dict with model_name, confidence_threshold, frame_interval, etc.

        Returns:
            Dictionary with detections
        """
        try:
            import cv2
            from ultralytics import YOLO

            device = self._get_device()
            model_name = config.get("model_name", "yolov8n.pt")
            confidence_threshold = config.get("confidence_threshold", 0.5)
            frame_interval_seconds = config.get("frame_interval", 1)

            logger.info(f"Object detection: {video_path} (device: {device})")

            # Open video and get properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")

            # Convert seconds to frame interval
            frame_interval = max(1, int(fps * frame_interval_seconds))
            frames_to_process = (total_frames + frame_interval - 1) // frame_interval
            logger.info(
                f"Processing every {frame_interval} frames "
                f"(every {frame_interval_seconds}s at {fps} FPS, "
                f"~{frames_to_process} frames to process)"
            )

            # Load model with explicit device
            model_path = str(self.cache_dir / "ultralytics" / model_name)
            model = YOLO(model_path)
            model.to(device)

            # Extract detections by reading only the frames we need
            detections = []
            frame_idx = 0

            while True:
                if frame_idx % frame_interval == 0:
                    # Read and decode frame for processing
                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp_ms = int((frame_idx / fps) * 1000)

                    # Run inference on single frame
                    results = model(
                        frame,
                        conf=confidence_threshold,
                        verbose=False,
                        device=device,
                    )

                    for result in results:
                        for box in result.boxes:
                            detection = {
                                "frame_index": frame_idx,
                                "timestamp_ms": timestamp_ms,
                                "label": result.names[int(box.cls)],
                                "confidence": float(box.conf),
                                "bbox": {
                                    "x": float(box.xyxy[0][0]),
                                    "y": float(box.xyxy[0][1]),
                                    "width": float(box.xyxy[0][2] - box.xyxy[0][0]),
                                    "height": float(box.xyxy[0][3] - box.xyxy[0][1]),
                                },
                            }
                            detections.append(detection)
                else:
                    # Skip frame without decoding (faster than read())
                    if not cap.grab():
                        break

                frame_idx += 1

            cap.release()

            logger.info(f"✅ Object detection complete: {len(detections)} detections")
            return {"detections": detections}

        except Exception as e:
            logger.error(f"Object detection failed: {e}", exc_info=True)
            raise

    async def detect_faces(self, video_path: str, config: dict) -> dict:
        """Detect faces in video using YOLO.

        Args:
            video_path: Path to video file
            config: Configuration dict with model_name, confidence_threshold, etc.
                   frame_interval can be in seconds (will be converted to frames)

        Returns:
            Dictionary with detections
        """
        try:
            import cv2
            from ultralytics import YOLO

            device = self._get_device()
            model_name = config.get("model_name", "yolov8n-face.pt")
            confidence_threshold = config.get("confidence_threshold", 0.7)
            frame_interval_seconds = config.get("frame_interval", 3)

            logger.info(f"Face detection: {video_path} (device: {device})")

            # Open video and get properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")

            # Convert seconds to frame interval
            frame_interval = max(1, int(fps * frame_interval_seconds))
            frames_to_process = (total_frames + frame_interval - 1) // frame_interval
            logger.info(
                f"Processing every {frame_interval} frames "
                f"(every {frame_interval_seconds}s at {fps} FPS, "
                f"~{frames_to_process} frames to process)"
            )

            # Load model with explicit device
            model_path = str(self.cache_dir / "ultralytics" / model_name)
            model = YOLO(model_path)
            model.to(device)

            # Extract detections by reading only the frames we need
            detections = []
            frame_idx = 0

            while True:
                if frame_idx % frame_interval == 0:
                    # Read and decode frame for processing
                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp_ms = int((frame_idx / fps) * 1000)

                    # Run inference on single frame
                    results = model(
                        frame,
                        conf=confidence_threshold,
                        verbose=False,
                        device=device,
                    )

                    for result in results:
                        for box in result.boxes:
                            confidence = float(box.conf)

                            # Additional safety filter: only keep high-confidence detections
                            if confidence < confidence_threshold:
                                continue

                            detection = {
                                "frame_index": frame_idx,
                                "timestamp_ms": timestamp_ms,
                                "label": "face",
                                "confidence": confidence,
                                "bbox": {
                                    "x": float(box.xyxy[0][0]),
                                    "y": float(box.xyxy[0][1]),
                                    "width": float(box.xyxy[0][2] - box.xyxy[0][0]),
                                    "height": float(box.xyxy[0][3] - box.xyxy[0][1]),
                                },
                                "cluster_id": None,
                            }
                            detections.append(detection)
                else:
                    # Skip frame without decoding (faster than read())
                    if not cap.grab():
                        break

                frame_idx += 1

            cap.release()

            logger.info(f"✅ Face detection complete: {len(detections)} detections")
            return {"detections": detections}

        except Exception as e:
            logger.error(f"Face detection failed: {e}", exc_info=True)
            raise

    async def transcribe_video(self, video_path: str, config: dict) -> dict:
        """Transcribe audio from video using Whisper.

        Args:
            video_path: Path to video file
            config: Configuration dict with model_name, languages, vad_filter, etc.

        Returns:
            Dictionary with segments
        """
        try:
            from faster_whisper import WhisperModel

            device = self._get_device()
            model_name = config.get("model_name", "base")
            languages = config.get("languages", None)
            # Handle both single language string and languages array
            language = None
            if languages:
                if isinstance(languages, list):
                    language = languages[0] if languages else None
                else:
                    language = languages
            vad_filter = config.get("vad_filter", True)

            logger.info(f"Transcription: {video_path} (device: {device})")

            # Load model with explicit device
            model = WhisperModel(model_name, device=device, compute_type="auto")

            # Run inference
            segments, info = model.transcribe(
                video_path,
                language=language,
                vad_filter=vad_filter,
            )

            # Extract segments
            transcription_segments = []
            detected_language = info.language
            for segment in segments:
                ts = {
                    "start_ms": int(segment.start * 1000),
                    "end_ms": int(segment.end * 1000),
                    "text": segment.text,
                    "language": detected_language,
                    "confidence": None,
                    "words": None,
                }
                transcription_segments.append(ts)

            logger.info(
                f"✅ Transcription complete: {len(transcription_segments)} segments"
            )
            return {"segments": transcription_segments}

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise

    async def extract_ocr(self, video_path: str, config: dict) -> dict:
        """Extract text from video frames using EasyOCR.

        Args:
            video_path: Path to video file
            config: Configuration dict with language or languages, frame_interval (in seconds), etc.
                   - language (str): Single language code (preferred)
                   - languages (list[str]): Legacy format, first language used if provided

        Returns:
            Dictionary with detections
        """
        try:
            import cv2
            import easyocr

            logger.info(f"OCR: {video_path} (GPU: {self.gpu_available})")

            # Handle both new 'language' (singular) and legacy 'languages' (list) formats
            language = config.get("language")
            if language:
                # New format: single language string
                languages = [language]
            else:
                # Legacy format: list of languages
                languages = config.get("languages", ["en"])
                if isinstance(languages, str):
                    languages = [languages]
                language = languages[0] if languages else "en"

            frame_interval_seconds = config.get("frame_interval", 2)

            # Load model with explicit GPU flag
            reader = easyocr.Reader(languages, gpu=self.gpu_available, verbose=False)

            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Convert seconds to frame interval
            frame_interval = max(1, int(fps * frame_interval_seconds))
            frames_to_process = (total_frames + frame_interval - 1) // frame_interval
            logger.info(
                f"Video FPS: {fps}, Total frames: {total_frames}, "
                f"Processing every {frame_interval} frames "
                f"(every {frame_interval_seconds}s, ~{frames_to_process} frames to process)"
            )

            frame_idx = 0
            detections = []

            while True:
                if frame_idx % frame_interval == 0:
                    # Read and decode frame for processing
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = reader.readtext(frame)
                    timestamp_ms = int((frame_idx / fps) * 1000)

                    for result in results:
                        bbox, text, confidence = result
                        detection = {
                            "frame_index": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "text": text,
                            "confidence": confidence,
                            "language": language,
                            "polygon": [
                                {"x": float(p[0]), "y": float(p[1])} for p in bbox
                            ],
                        }
                        detections.append(detection)
                else:
                    # Skip frame without decoding (faster than read())
                    if not cap.grab():
                        break

                frame_idx += 1

            cap.release()

            logger.info(f"✅ OCR complete: {len(detections)} detections")
            return {"detections": detections, "language": language}

        except Exception as e:
            logger.error(f"OCR failed: {e}", exc_info=True)
            raise

    async def classify_places(self, video_path: str, config: dict) -> dict:
        """Classify places in video frames using Places365.

        Args:
            video_path: Path to video file
            config: Configuration dict with frame_interval (in seconds), top_k, etc.

        Returns:
            Dictionary with classifications
        """
        try:
            import cv2
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            from PIL import Image

            device = self._get_device()
            logger.info(f"Place detection: {video_path} (device: {device})")

            # Load Places365 labels
            # Try multiple locations: cache dir, project root, or relative to module
            possible_paths = [
                self.cache_dir / "places365" / "categories_places365.txt",
                Path(__file__).parent.parent.parent / "categories_places365.txt",
                (
                    Path(__file__).parent.parent.parent.parent
                    / "ml-service"
                    / "categories_places365.txt"
                ),
            ]

            labels_path = None
            for path in possible_paths:
                if path.exists():
                    labels_path = path
                    break

            if labels_path is None:
                logger.warning(
                    "Places365 labels not found in any location, using generic labels"
                )
                classes = [f"place_{i}" for i in range(365)]
            else:
                logger.info(f"Loading Places365 labels from {labels_path}")
                with open(labels_path) as f:
                    classes = [line.strip().split(" ")[0][3:] for line in f.readlines()]

            # Load Places365 model
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 365)

            # Try to load pretrained weights if available
            model_path = self.cache_dir / "places365" / "resnet18_places365.pth.tar"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                # Remove 'module.' prefix from keys if present
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("module.", "")
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)

            model.to(device)
            model.eval()

            # Prepare transforms
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            # Process video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval_seconds = config.get("frame_interval", 1)
            top_k = config.get("top_k", 5)

            # Convert seconds to frame interval
            frame_interval = max(1, int(fps * frame_interval_seconds))
            frames_to_process = (total_frames + frame_interval - 1) // frame_interval
            logger.info(
                f"Video FPS: {fps}, Total frames: {total_frames}, "
                f"Processing every {frame_interval} frames "
                f"(every {frame_interval_seconds}s, ~{frames_to_process} frames to process)"
            )

            classifications = []
            frame_idx = 0

            while True:
                if frame_idx % frame_interval == 0:
                    # Read and decode frame for processing
                    ret, frame = cap.read()
                    if not ret:
                        break

                    try:
                        # Convert frame to PIL Image
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        input_img = transform(img).unsqueeze(0).to(device)

                        # Run inference
                        with torch.no_grad():
                            logit = model(input_img)
                            h_x = torch.nn.functional.softmax(logit, 1).squeeze()
                            probs, idx = h_x.sort(0, True)

                        # Extract top-k predictions
                        timestamp_ms = int((frame_idx / fps) * 1000)
                        top_predictions = [
                            {
                                "label": classes[int(i)],
                                "confidence": float(probs[j]),
                            }
                            for j, i in enumerate(idx[:top_k])
                        ]

                        classification = {
                            "frame_index": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "predictions": top_predictions,
                        }
                        classifications.append(classification)

                    except Exception as e:
                        logger.warning(f"Error classifying frame {frame_idx}: {e}")
                else:
                    # Skip frame without decoding (faster than read())
                    if not cap.grab():
                        break

                frame_idx += 1

            cap.release()

            logger.info(
                f"✅ Place detection complete: {len(classifications)} classifications"
            )
            return {"classifications": classifications}

        except Exception as e:
            logger.error(f"Place detection failed: {e}", exc_info=True)
            raise

    async def detect_scenes(self, video_path: str, config: dict) -> dict:
        """Detect scene boundaries in video using ffmpeg.

        Args:
            video_path: Path to video file
            config: Configuration dict with threshold, min_scene_length, etc.

        Returns:
            Dictionary with scenes
        """
        try:
            import subprocess

            logger.info(f"Scene detection: {video_path}")

            # Get configuration
            # Threshold is 0-1 scale where higher = fewer scene cuts
            threshold = config.get("threshold", 0.7)

            # Use ffmpeg with scene detection filter
            # The scenecut filter detects scene changes based on frame differences
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                f"select='gt(scene\\,{threshold})',showinfo",
                "-f",
                "null",
                "-",
            ]

            logger.info(f"Running ffmpeg scene detection with threshold {threshold}")

            # Run ffmpeg and capture stderr (where showinfo outputs)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            # Parse ffmpeg output for scene changes
            scenes = []
            scene_idx = 0
            prev_timestamp_ms = 0

            for line in result.stderr.split("\n"):
                if "showinfo" in line and "pts_time:" in line:
                    # Extract timestamp from showinfo output
                    # Format: ... pts_time:123.456 ...
                    try:
                        parts = line.split("pts_time:")
                        if len(parts) > 1:
                            timestamp_str = parts[1].split()[0]
                            timestamp_s = float(timestamp_str)
                            timestamp_ms = int(timestamp_s * 1000)

                            # Create scene from previous timestamp to current
                            if scene_idx > 0:
                                scene = {
                                    "scene_index": scene_idx - 1,
                                    "start_ms": prev_timestamp_ms,
                                    "end_ms": timestamp_ms,
                                    "duration_ms": timestamp_ms - prev_timestamp_ms,
                                }
                                scenes.append(scene)

                            prev_timestamp_ms = timestamp_ms
                            scene_idx += 1
                    except (ValueError, IndexError):
                        continue

            # Get video duration for final scene or fallback
            duration_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1:nokey=1",
                video_path,
            ]
            duration_result = subprocess.run(
                duration_cmd, capture_output=True, text=True, timeout=60
            )
            try:
                duration_ms = int(float(duration_result.stdout.strip()) * 1000)
            except (ValueError, IndexError):
                duration_ms = prev_timestamp_ms + 1000

            # Add final scene if we detected any changes
            if scene_idx > 0:
                scene = {
                    "scene_index": scene_idx,
                    "start_ms": prev_timestamp_ms,
                    "end_ms": duration_ms,
                    "duration_ms": duration_ms - prev_timestamp_ms,
                }
                scenes.append(scene)
            else:
                # No scene cuts detected - create one scene for entire video
                scene = {
                    "scene_index": 0,
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "duration_ms": duration_ms,
                }
                scenes.append(scene)
                logger.info(
                    f"No scene cuts detected. Created single scene for entire video "
                    f"({duration_ms}ms)"
                )

            logger.info(f"✅ Scene detection complete: {len(scenes)} scenes")
            return {"scenes": scenes}

        except Exception as e:
            logger.error(f"Scene detection failed: {e}", exc_info=True)
            raise
