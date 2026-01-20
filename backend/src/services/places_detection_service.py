"""Places detection service using ResNet18 Places365."""

import uuid
from pathlib import Path

import av
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..domain.models import Place
from ..utils.print_logger import get_logger

logger = get_logger(__name__)


class PlacesDetectionError(Exception):
    """Exception raised for places detection errors."""

    pass


class PlacesDetectionService:
    """Service for detecting places/scenes in video frames using ResNet18 Places365."""

    def __init__(self, model_path: str = "models/resnet18_places365.pth.tar"):
        """Initialize places detection service.

        Args:
            model_path: Path to ResNet18 Places365 model file
        """
        self.model_path = model_path
        self.model = None
        self.classes = []
        self.transform = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize ResNet18 Places365 model."""
        try:
            logger.info(f"Loading Places365 model: {self.model_path}")

            # Load class labels
            labels_path = "categories_places365.txt"
            if Path(labels_path).exists():
                with open(labels_path) as f:
                    self.classes = [
                        line.strip().split(" ")[0][3:] for line in f.readlines()
                    ]
            else:
                raise PlacesDetectionError(
                    f"Places365 labels file not found: {labels_path}"
                )

            # Load ResNet18 model
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet18", pretrained=False
            )

            # Load Places365 weights
            if not Path(self.model_path).exists():
                raise PlacesDetectionError(
                    f"Places365 model file not found: {self.model_path}"
                )

            checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
            state_dict = checkpoint["state_dict"]

            # Remove 'module.' prefix from keys if present
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v

            # Modify final layer for 365 classes
            model.fc = torch.nn.Linear(model.fc.in_features, 365)
            model.load_state_dict(new_state_dict)
            model.eval()

            self.model = model

            # Define image preprocessing
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            logger.info("Places365 model loaded successfully")

        except Exception as e:
            raise PlacesDetectionError(f"Failed to load Places365 model: {e}") from e

    def detect_places_in_video(
        self, video_path: str, video_id: str, sample_rate: int = 30, top_k: int = 5
    ) -> list[Place]:
        """Detect places/scenes in video frames.

        Args:
            video_path: Path to video file
            video_id: Video ID for associating detections
            sample_rate: Process every Nth frame (default: 30 frames)
            top_k: Number of top predictions to store (default: 5)

        Returns:
            List of Place domain models with detections
        """
        if not Path(video_path).exists():
            raise PlacesDetectionError(f"Video file not found: {video_path}")

        logger.info(f"Starting places detection for video: {video_path}")
        logger.info(f"Sample rate: every {sample_rate} frames, top-{top_k} predictions")

        try:
            container = av.open(video_path)
        except av.AVError as e:
            raise PlacesDetectionError(f"Failed to open video: {e}") from e

        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)

        logger.info(
            f"Video info: {video_stream.codec_context.name} codec, "
            f"{fps:.2f} fps, {video_stream.frames} frames"
        )

        frame_idx = 0
        processed_frames = 0

        # Dictionary to aggregate detections by label
        # label -> {timestamps: [], confidences: [], alternative_labels: []}
        detections_by_label = {}

        try:
            for frame in container.decode(video=0):
                # Sample frames based on sample_rate
                if frame_idx % sample_rate == 0:
                    # Convert PyAV frame to PIL Image
                    img = frame.to_image()

                    timestamp = frame_idx / fps if fps > 0 else frame_idx
                    self._process_frame(
                        img, timestamp, detections_by_label, frame_idx, top_k
                    )
                    processed_frames += 1

                    if processed_frames % 100 == 0:
                        logger.debug(
                            f"Processed {processed_frames} frames, "
                            f"found {len(detections_by_label)} unique places"
                        )

                frame_idx += 1

        finally:
            container.close()

        logger.info(
            f"Places detection complete. Processed {processed_frames} frames, "
            f"found {len(detections_by_label)} unique place labels"
        )

        # Convert aggregated detections to Place domain models
        places = []
        for label, data in detections_by_label.items():
            # Calculate average confidence
            avg_confidence = (
                sum(data["confidences"]) / len(data["confidences"])
                if data["confidences"]
                else 0.0
            )

            # Get most common alternative labels
            alternative_labels = self._aggregate_alternative_labels(
                data["alternative_labels"], top_k
            )

            place = Place(
                place_id=str(uuid.uuid4()),
                video_id=video_id,
                label=label,
                timestamps=data["timestamps"],
                confidence=avg_confidence,
                alternative_labels=alternative_labels,
                metadata={"model_name": "resnet18_places365", "top_k": top_k},
            )
            places.append(place)

        return places

    def _process_frame(
        self,
        frame: Image.Image,
        timestamp: float,
        detections_by_label: dict,
        frame_idx: int,
        top_k: int,
    ):
        """Process a single frame and aggregate detections.

        Args:
            frame: PIL Image
            timestamp: Timestamp in seconds
            detections_by_label: Dictionary to aggregate detections
            frame_idx: Frame index for logging
            top_k: Number of top predictions to store
        """
        try:
            # Preprocess image
            input_img = self.transform(frame).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                logit = self.model(input_img)
                h_x = torch.nn.functional.softmax(logit, 1).squeeze()
                probs, idx = h_x.sort(0, True)

            # Get top-K predictions
            top_predictions = [
                {"label": self.classes[i], "confidence": float(probs[i])}
                for i in idx[:top_k]
            ]

            # Use top prediction as dominant label
            dominant_label = top_predictions[0]["label"]
            dominant_confidence = top_predictions[0]["confidence"]

            # Initialize label entry if first occurrence
            if dominant_label not in detections_by_label:
                detections_by_label[dominant_label] = {
                    "timestamps": [],
                    "confidences": [],
                    "alternative_labels": [],
                }

            # Add detection
            detections_by_label[dominant_label]["timestamps"].append(timestamp)
            detections_by_label[dominant_label]["confidences"].append(
                dominant_confidence
            )
            detections_by_label[dominant_label]["alternative_labels"].append(
                top_predictions[1:]  # Store alternatives (excluding dominant)
            )

        except Exception as e:
            logger.warning(f"Error processing frame {frame_idx}: {e}")

    def _aggregate_alternative_labels(
        self, alternative_labels_list: list, top_k: int
    ) -> list[dict]:
        """Aggregate alternative labels across all detections.

        Args:
            alternative_labels_list: List of alternative label lists
            top_k: Number of top alternatives to return

        Returns:
            List of aggregated alternative labels with average confidence
        """
        # Count occurrences and sum confidences
        label_stats = {}
        for alternatives in alternative_labels_list:
            for alt in alternatives:
                label = alt["label"]
                confidence = alt["confidence"]
                if label not in label_stats:
                    label_stats[label] = {"count": 0, "total_confidence": 0.0}
                label_stats[label]["count"] += 1
                label_stats[label]["total_confidence"] += confidence

        # Calculate average confidence and sort
        aggregated = []
        for label, stats in label_stats.items():
            avg_confidence = stats["total_confidence"] / stats["count"]
            aggregated.append({"label": label, "confidence": avg_confidence})

        # Sort by confidence and return top-K
        aggregated.sort(key=lambda x: x["confidence"], reverse=True)
        return aggregated[:top_k]
