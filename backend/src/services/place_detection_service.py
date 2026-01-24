"""Place classification service using ResNet18 Places365 model."""

import logging
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


class PlaceDetectionService:
    """Service for detecting places/scenes in video frames using ResNet18 Places365."""

    def __init__(
        self,
        model_path: str | None = None,
        labels_path: str | None = None,
    ):
        """Initialize the place detection service.

        Args:
            model_path: Path to the ResNet18 Places365 model file
            labels_path: Path to the Places365 categories file
        """
        # Use paths relative to the backend directory
        backend_dir = Path(__file__).parent.parent.parent

        if model_path is None:
            model_path = str(backend_dir / "resnet18_places365.pth.tar")
        if labels_path is None:
            labels_path = str(backend_dir / "categories_places365.txt")

        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.classes = None
        self.transform = None
        self._load_model()

    def _load_model(self):
        """Load the ResNet18 Places365 model and labels."""
        try:
            # Load labels
            with open(self.labels_path) as f:
                # Format: /a/abbey 0
                # We want just "abbey"
                self.classes = [
                    line.strip().split(" ")[0][3:] for line in f.readlines()
                ]

            logger.info(f"Loaded {len(self.classes)} place categories")

            # Load model architecture
            self.model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet18", pretrained=False
            )

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
            state_dict = checkpoint["state_dict"]

            # Remove 'module.' prefix from keys if present
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v

            # Adjust final layer for 365 classes
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 365)
            self.model.load_state_dict(new_state_dict)
            self.model.eval()

            # Define image transformation
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            logger.info("ResNet18 Places365 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Places365 model: {e}")
            raise

    def classify_frame(self, frame, top_k: int = 5) -> list[dict]:
        """Classify a single frame.

        Args:
            frame: OpenCV frame (BGR format)
            top_k: Number of top predictions to return

        Returns:
            List of dicts with 'label' and 'confidence' keys, sorted by confidence
        """
        try:
            # Convert BGR to RGB
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Transform and add batch dimension
            input_img = self.transform(img).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                logit = self.model(input_img)
                h_x = torch.nn.functional.softmax(logit, 1).squeeze()
                probs, idx = h_x.sort(0, True)

                # Return top_k predictions
                results = []
                for i in range(min(top_k, len(idx))):
                    results.append(
                        {"label": self.classes[idx[i]], "confidence": float(probs[i])}
                    )

                return results

        except Exception as e:
            logger.error(f"Failed to classify frame: {e}")
            return []

    def detect_places_in_video(
        self, video_path: str, sample_rate: int = 30, top_k: int = 5
    ) -> list[dict]:
        """Detect places in a video by sampling frames.

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame
            top_k: Number of top predictions per frame

        Returns:
            List of dicts with frame_number, timestamp, and classifications
        """
        results = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return results

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames at specified rate
                if frame_idx % sample_rate == 0:
                    timestamp_sec = frame_idx / fps
                    classifications = self.classify_frame(frame, top_k=top_k)

                    if classifications:
                        results.append(
                            {
                                "frame_number": frame_idx,
                                "timestamp": timestamp_sec,
                                "classifications": classifications,
                            }
                        )

                frame_idx += 1

            cap.release()
            logger.info(
                f"Processed {len(results)} frames from video {Path(video_path).name}"
            )

        except Exception as e:
            logger.error(f"Failed to process video {video_path}: {e}")

        return results
