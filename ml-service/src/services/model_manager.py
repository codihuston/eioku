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

    async def download_model(self, model_name: str, model_type: str) -> Path:
        """Download model from source and cache locally.

        Args:
            model_name: Name of the model to download
            model_type: Type of model (yolo, whisper, places365, easyocr)

        Returns:
            Path to cached model
        """
        cache_path = self.cache_dir / model_name

        if cache_path.exists():
            logger.info(f"Model {model_name} already cached at {cache_path}")
            return cache_path

        logger.info(f"Downloading {model_type} model: {model_name}")

        try:
            if model_type == "yolo":
                from ultralytics import YOLO

                model = YOLO(model_name)
                model.save(str(cache_path))

            elif model_type == "whisper":
                # faster-whisper handles caching internally
                from faster_whisper import WhisperModel

                WhisperModel(model_name, device="auto", compute_type="auto")
                logger.info(f"Whisper model {model_name} cached by faster-whisper")

            elif model_type == "easyocr":
                import easyocr

                easyocr.Reader(["en"], gpu=torch.cuda.is_available())
                logger.info(f"EasyOCR model cached")

            logger.info(f"✓ Downloaded {model_name}")
            return cache_path

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
                logger.info(f"✓ YOLO model {model_name} verified")

            elif model_type == "whisper":
                from faster_whisper import WhisperModel

                WhisperModel(model_name, device="auto", compute_type="auto")
                logger.info(f"✓ Whisper model {model_name} verified")

            elif model_type == "easyocr":
                import easyocr

                easyocr.Reader(["en"], gpu=torch.cuda.is_available())
                logger.info(f"✓ EasyOCR model verified")

            # Log GPU detection result
            if torch.cuda.is_available():
                logger.info(f"  GPU detected: {torch.cuda.get_device_name(0)}")
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
        if not torch.cuda.is_available():
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
