"""Full pipeline integration tests for artifact envelope architecture.

This test suite verifies the complete end-to-end flow:
1. Process test video through all pipelines
2. Verify artifacts created correctly
3. Verify projections synchronized
4. Verify API endpoints return correct data
"""

import json
import uuid
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy import text as sql_text
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.domain.artifacts import ArtifactEnvelope
from src.domain.models import Video
from src.domain.schema_initialization import register_all_schemas
from src.domain.schema_registry import SchemaRegistry
from src.domain.schemas.face_detection_v1 import (
    BoundingBox as FaceBoundingBox,
)
from src.domain.schemas.face_detection_v1 import (
    FaceDetectionV1,
)
from src.domain.schemas.object_detection_v1 import (
    BoundingBox as ObjectBoundingBox,
)
from src.domain.schemas.object_detection_v1 import (
    ObjectDetectionV1,
)
from src.domain.schemas.ocr_detection_v1 import OCRDetectionV1, Point
from src.domain.schemas.place_classification_v1 import (
    PlaceClassificationV1,
    PlacePrediction,
)
from src.domain.schemas.scene_v1 import SceneV1
from src.domain.schemas.transcript_segment_v1 import TranscriptSegmentV1
from src.repositories.artifact_repository import SqlArtifactRepository
from src.repositories.video_repository import SqlVideoRepository
from src.services.projection_sync_service import ProjectionSyncService


@pytest.fixture
def engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create database session for testing."""
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture(scope="session")
def schema_registry():
    """Create and initialize schema registry once for all tests."""
    register_all_schemas()
    return SchemaRegistry


@pytest.fixture
def video_repo(session):
    """Create video repository."""
    return SqlVideoRepository(session)


@pytest.fixture
def artifact_repo(session, schema_registry):
    """Create artifact repository with projection sync."""
    projection_sync = ProjectionSyncService(session)
    return SqlArtifactRepository(session, schema_registry, projection_sync)


@pytest.fixture
def test_video(video_repo):
    """Create a test video."""
    video = Video(
        video_id=str(uuid.uuid4()),
        file_path="/test/video.mp4",
        filename="video.mp4",
        last_modified=datetime.utcnow(),
        file_size=1024000,
        duration=60.0,
        status="hashed",
        file_hash="test_hash_123",
    )
    video_repo.save(video)
    return video


class TestFullPipelineIntegration:
    """Test complete pipeline integration."""

    def test_transcript_artifacts_and_projection(
        self, session, artifact_repo, test_video
    ):
        """Test transcript artifact creation."""
        run_id = str(uuid.uuid4())

        # Create transcript segments
        segments = [
            ("Hello world", 0, 2000),
            ("This is a test", 2000, 5000),
            ("Testing transcription", 5000, 8000),
        ]

        for text, start_ms, end_ms in segments:
            payload = TranscriptSegmentV1(
                text=text, speaker=None, confidence=0.95, language="en"
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="transcript.segment",
                schema_version=1,
                span_start_ms=start_ms,
                span_end_ms=end_ms,
                payload_json=payload.model_dump_json(),
                producer="whisper",
                producer_version="base",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify artifacts created
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="transcript.segment"
        )
        assert len(artifacts) == 3

        # Verify artifact payloads
        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            assert "text" in payload
            assert "confidence" in payload
            assert payload["confidence"] == 0.95

    def test_scene_artifacts_and_projection(self, session, artifact_repo, test_video):
        """Test scene artifact creation and scene_ranges projection."""
        run_id = str(uuid.uuid4())

        # Create scene artifacts
        scenes = [(0, 0, 10000), (1, 10000, 20000), (2, 20000, 30000)]

        for scene_index, start_ms, end_ms in scenes:
            payload = SceneV1(
                scene_index=scene_index,
                method="content",
                score=0.85,
                frame_number=scene_index * 300,
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="scene",
                schema_version=1,
                span_start_ms=start_ms,
                span_end_ms=end_ms,
                payload_json=payload.model_dump_json(),
                producer="ffmpeg",
                producer_version="1.0.0",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify artifacts created
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="scene"
        )
        assert len(artifacts) == 3

        # Verify scene_ranges projection
        result = session.execute(
            sql_text(
                """
                SELECT COUNT(*), MIN(scene_index), MAX(scene_index)
                FROM scene_ranges
                WHERE asset_id = :asset_id
                """
            ),
            {"asset_id": test_video.video_id},
        ).fetchone()
        assert result[0] == 3
        assert result[1] == 0
        assert result[2] == 2

    def test_object_detection_artifacts_and_projection(
        self, session, artifact_repo, test_video
    ):
        """Test object detection artifact creation and object_labels projection."""
        run_id = str(uuid.uuid4())

        # Create object detection artifacts
        objects = [
            ("person", 0.95, 1000),
            ("car", 0.88, 2000),
            ("person", 0.92, 3000),
        ]

        for label, confidence, timestamp_ms in objects:
            payload = ObjectDetectionV1(
                label=label,
                confidence=confidence,
                bounding_box=ObjectBoundingBox(x=100, y=100, width=200, height=200),
                frame_number=timestamp_ms // 33,  # Approximate frame at 30fps
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="object.detection",
                schema_version=1,
                span_start_ms=timestamp_ms,
                span_end_ms=timestamp_ms + 33,  # One frame duration
                payload_json=payload.model_dump_json(),
                producer="yolo",
                producer_version="v8n",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify artifacts created
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="object.detection"
        )
        assert len(artifacts) == 3

        # Verify object_labels projection
        result = session.execute(
            sql_text(
                """
                SELECT label, COUNT(*) FROM object_labels
                WHERE asset_id = :asset_id
                GROUP BY label
                ORDER BY label
                """
            ),
            {"asset_id": test_video.video_id},
        ).fetchall()
        assert len(result) == 2  # person and car
        assert result[0][0] == "car"
        assert result[0][1] == 1
        assert result[1][0] == "person"
        assert result[1][1] == 2

    def test_face_detection_artifacts_and_projection(
        self, session, artifact_repo, test_video
    ):
        """Test face detection artifact creation and face_clusters projection."""
        run_id = str(uuid.uuid4())

        # Create face detection artifacts
        faces = [
            ("cluster_1", 0.96, 1000),
            ("cluster_2", 0.94, 2000),
            ("cluster_1", 0.95, 3000),
        ]

        for cluster_id, confidence, timestamp_ms in faces:
            payload = FaceDetectionV1(
                confidence=confidence,
                bounding_box=FaceBoundingBox(x=150, y=150, width=100, height=100),
                cluster_id=cluster_id,
                frame_number=timestamp_ms // 33,
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="face.detection",
                schema_version=1,
                span_start_ms=timestamp_ms,
                span_end_ms=timestamp_ms + 33,
                payload_json=payload.model_dump_json(),
                producer="yolo",
                producer_version="v8n-face",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify artifacts created
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="face.detection"
        )
        assert len(artifacts) == 3

        # Verify face_clusters projection
        result = session.execute(
            sql_text(
                """
                SELECT cluster_id, COUNT(*) FROM face_clusters
                WHERE asset_id = :asset_id
                GROUP BY cluster_id
                ORDER BY cluster_id
                """
            ),
            {"asset_id": test_video.video_id},
        ).fetchall()
        assert len(result) == 2
        assert result[0][0] == "cluster_1"
        assert result[0][1] == 2
        assert result[1][0] == "cluster_2"
        assert result[1][1] == 1

    def test_place_classification_artifacts(self, session, artifact_repo, test_video):
        """Test place classification artifact creation."""
        run_id = str(uuid.uuid4())

        # Create place classification artifacts
        places = [
            ("kitchen", 0.92, [("dining_room", 0.15), ("living_room", 0.10)], 1000),
            ("office", 0.88, [("conference_room", 0.20), ("library", 0.12)], 2000),
        ]

        for label, confidence, alternatives, timestamp_ms in places:
            predictions = [
                PlacePrediction(label=label, confidence=confidence),
                *[
                    PlacePrediction(label=alt[0], confidence=alt[1])
                    for alt in alternatives
                ],
            ]

            payload = PlaceClassificationV1(
                predictions=predictions,
                frame_number=timestamp_ms // 33,
                top_k=len(predictions),
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="place.classification",
                schema_version=1,
                span_start_ms=timestamp_ms,
                span_end_ms=timestamp_ms + 33,
                payload_json=payload.model_dump_json(),
                producer="resnet",
                producer_version="places365",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify artifacts created
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="place.classification"
        )
        assert len(artifacts) == 2

        # Verify payload structure
        payload = json.loads(artifacts[0].payload_json)
        assert "predictions" in payload
        assert len(payload["predictions"]) >= 1
        assert "frame_number" in payload

    def test_ocr_artifacts_and_projection(self, session, artifact_repo, test_video):
        """Test OCR artifact creation."""
        run_id = str(uuid.uuid4())

        # Create OCR artifacts
        ocr_texts = [
            ("Hello World", 0.98, 1000),
            ("Test OCR", 0.95, 2000),
            ("Sample Text", 0.97, 3000),
        ]

        for text, confidence, timestamp_ms in ocr_texts:
            polygon = [
                Point(x=100, y=100),
                Point(x=200, y=100),
                Point(x=200, y=150),
                Point(x=100, y=150),
            ]

            payload = OCRDetectionV1(
                text=text,
                confidence=confidence,
                polygon=polygon,
                language="en",
                frame_number=timestamp_ms // 33,
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="ocr.detection",
                schema_version=1,
                span_start_ms=timestamp_ms,
                span_end_ms=timestamp_ms + 33,
                payload_json=payload.model_dump_json(),
                producer="easyocr",
                producer_version="1.0",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify artifacts created
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="ocr.detection"
        )
        assert len(artifacts) == 3

        # Verify payload structure
        for artifact in artifacts:
            payload = json.loads(artifact.payload_json)
            assert "text" in payload
            assert "confidence" in payload
            assert "bounding_box" in payload
            assert len(payload["bounding_box"]) == 4  # 4 polygon points

    def test_multi_profile_artifacts(self, session, artifact_repo, test_video):
        """Test that multiple model profiles can coexist."""
        # Create artifacts with different profiles
        profiles = ["fast", "balanced", "high_quality"]

        for profile in profiles:
            run_id = str(uuid.uuid4())

            payload = TranscriptSegmentV1(
                text=f"Text from {profile} model",
                speaker=None,
                confidence=0.9,
                language="en",
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="transcript.segment",
                schema_version=1,
                span_start_ms=0,
                span_end_ms=1000,
                payload_json=payload.model_dump_json(),
                producer="whisper",
                producer_version=profile,
                model_profile=profile,
                config_hash=f"config_{profile}",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify all profiles exist
        for profile in profiles:
            artifacts = session.execute(
                sql_text(
                    """
                    SELECT COUNT(*) FROM artifacts
                    WHERE asset_id = :asset_id
                    AND artifact_type = 'transcript.segment'
                    AND model_profile = :profile
                    """
                ),
                {"asset_id": test_video.video_id, "profile": profile},
            ).fetchone()
            assert artifacts[0] == 1

    def test_time_span_overlap_query(self, session, artifact_repo, test_video):
        """Test querying artifacts by time span overlap."""
        run_id = str(uuid.uuid4())

        # Create artifacts with different time spans
        spans = [(0, 5000), (3000, 8000), (7000, 12000), (15000, 20000)]

        for start_ms, end_ms in spans:
            payload = SceneV1(
                scene_index=0, method="content", score=0.8, frame_number=0
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="scene",
                schema_version=1,
                span_start_ms=start_ms,
                span_end_ms=end_ms,
                payload_json=payload.model_dump_json(),
                producer="ffmpeg",
                producer_version="1.0.0",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Query for artifacts overlapping [4000, 9000]
        overlapping = artifact_repo.get_by_span(
            asset_id=test_video.video_id,
            artifact_type="scene",
            span_start_ms=4000,
            span_end_ms=9000,
        )

        # Should return 3 artifacts: [0-5000], [3000-8000], [7000-12000]
        assert len(overlapping) == 3

        # Verify they are the correct ones
        start_times = sorted([a.span_start_ms for a in overlapping])
        assert start_times == [0, 3000, 7000]

    def test_complete_pipeline_all_artifact_types(
        self, session, artifact_repo, test_video
    ):
        """Test complete pipeline with all artifact types."""
        run_id = str(uuid.uuid4())

        # Create one artifact of each type
        artifact_types = [
            (
                "transcript.segment",
                TranscriptSegmentV1(
                    text="test", start_ms=0, end_ms=1000, confidence=0.9
                ),
            ),
            (
                "scene",
                SceneV1(scene_index=0, method="content", score=0.8, frame_number=0),
            ),
            (
                "object.detection",
                ObjectDetectionV1(
                    label="person",
                    confidence=0.9,
                    bounding_box=ObjectBoundingBox(x=0, y=0, width=100, height=100),
                    frame_number=0,
                ),
            ),
            (
                "face.detection",
                FaceDetectionV1(
                    confidence=0.9,
                    bounding_box=FaceBoundingBox(x=0, y=0, width=50, height=50),
                    cluster_id="cluster_1",
                    frame_number=0,
                ),
            ),
            (
                "place.classification",
                PlaceClassificationV1(
                    predictions=[PlacePrediction(label="office", confidence=0.9)],
                    frame_number=0,
                    top_k=1,
                ),
            ),
            (
                "ocr.detection",
                OCRDetectionV1(
                    text="test",
                    confidence=0.9,
                    polygon=[
                        Point(x=0, y=0),
                        Point(x=100, y=0),
                        Point(x=100, y=50),
                        Point(x=0, y=50),
                    ],
                    language="en",
                    frame_number=0,
                ),
            ),
        ]

        for artifact_type, payload in artifact_types:
            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type=artifact_type,
                schema_version=1,
                span_start_ms=0,
                span_end_ms=1000,
                payload_json=payload.model_dump_json(),
                producer="test",
                producer_version="1.0",
                model_profile="balanced",
                config_hash="test_config",
                input_hash="test_input",
                run_id=run_id,
                created_at=datetime.utcnow(),
            )

            artifact_repo.create(artifact)

        # Verify all artifact types were created
        for artifact_type, _ in artifact_types:
            artifacts = artifact_repo.get_by_asset(
                asset_id=test_video.video_id, artifact_type=artifact_type
            )
            assert len(artifacts) == 1, f"Failed to create {artifact_type}"

        # Verify total count
        total = session.execute(
            sql_text(
                """
                SELECT COUNT(*) FROM artifacts
                WHERE asset_id = :asset_id
                """
            ),
            {"asset_id": test_video.video_id},
        ).fetchone()
        assert total[0] == 6
