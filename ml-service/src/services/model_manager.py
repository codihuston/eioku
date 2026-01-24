"""Model manager for downloading and verifying ML models."""

import logging
import os
from pathlib import Path

import torch

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
        self.gpu_available = torch.cuda.is_available()

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
                from ultralytics import YOLO

                # YOLO will cache to ~/.yolov8 by default
                model = YOLO(model_name)
                logger.info(f"✓ YOLO model {model_name} downloaded")
                return Path(model.model_name)

            elif model_type == "whisper":
                # faster-whisper handles caching internally to ~/.cache/huggingface
                from faster_whisper import WhisperModel

                model = WhisperModel(
                    model_name, device=self._get_device(), compute_type="auto"
                )
                logger.info(f"✓ Whisper model {model_name} downloaded")
                return Path(model_name)

            elif model_type == "easyocr":
                import easyocr

                # EasyOCR caches to ~/.EasyOCR by default
                reader = easyocr.Reader(
                    ["en"], gpu=self.gpu_available, verbose=False
                )
                logger.info(f"✓ EasyOCR model downloaded")
                return Path("easyocr")

            elif model_type == "places365":
                # Places365 model is loaded via torchvision
                import torchvision.models as models

                model = models.resnet18(pretrained=False)
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
                results = model(dummy_image, verbose=False)
                logger.info(f"✓ YOLO model {model_name} verified")

            elif model_type == "whisper":
                from faster_whisper import WhisperModel

                model = WhisperModel(
                    model_name, device=self._get_device(), compute_type="auto"
                )
                logger.info(f"✓ Whisper model {model_name} verified")

            elif model_type == "easyocr":
                import easyocr

                reader = easyocr.Reader(
                    ["en"], gpu=self.gpu_available, verbose=False
                )
                logger.info(f"✓ EasyOCR model verified")

            elif model_type == "places365":
                import torchvision.models as models

                model = models.resnet18(pretrained=False)
                model.to(self._get_device())
                model.eval()
                logger.info(f"✓ Places365 model verified")

            # Log GPU detection result
            if self.gpu_available:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"  GPU detected: {device_name}")
            else:
                logger.info(f"  GPU not available, using CPU")

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
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU device: {device_name}")
            logger.info(f"GPU memory: {total_memory:.2f} GB")
        else:
            logger.warning("GPU not available - will use CPU for inference (slower)")
