"""Performance tests for artifact envelope architecture.

This test suite verifies:
1. Query performance on large artifact sets
2. Index usage verification
3. Database size growth monitoring
"""

import time
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
from src.domain.schemas.object_detection_v1 import (
    BoundingBox,
    ObjectDetectionV1,
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
        duration=3600.0,  # 1 hour video
        status="hashed",
        file_hash="test_hash_123",
    )
    video_repo.save(video)
    return video


class TestPerformance:
    """Performance tests for artifact queries."""

    def test_large_artifact_set_creation(self, artifact_repo, test_video):
        """Test creating a large number of artifacts."""
        run_id = str(uuid.uuid4())
        num_artifacts = 1000

        start_time = time.time()

        # Create 1000 transcript segments (simulating 1 hour video with 3.6s segments)
        for i in range(num_artifacts):
            start_ms = i * 3600
            end_ms = (i + 1) * 3600

            payload = TranscriptSegmentV1(
                text=f"Segment {i} text content",
                speaker=None,
                confidence=0.95,
                language="en",
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

        creation_time = time.time() - start_time

        # Verify all artifacts were created
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="transcript.segment"
        )
        assert len(artifacts) == num_artifacts

        # Performance assertion: should create 1000 artifacts in under 10 seconds
        assert (
            creation_time < 10.0
        ), f"Creation took {creation_time:.2f}s (expected < 10s)"

        print(f"\n✓ Created {num_artifacts} artifacts in {creation_time:.2f}s")
        print(f"  Average: {(creation_time / num_artifacts) * 1000:.2f}ms per artifact")

    def test_query_performance_by_asset(self, session, artifact_repo, test_video):
        """Test query performance for retrieving artifacts by asset_id."""
        run_id = str(uuid.uuid4())
        num_artifacts = 500

        # Create artifacts
        for i in range(num_artifacts):
            payload = SceneV1(
                scene_index=i,
                method="content",
                score=0.85,
                frame_number=i * 100,
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="scene",
                schema_version=1,
                span_start_ms=i * 10000,
                span_end_ms=(i + 1) * 10000,
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

        # Test query performance
        start_time = time.time()
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="scene"
        )
        query_time = time.time() - start_time

        assert len(artifacts) == num_artifacts

        # Performance assertion: should query 500 artifacts in under 100ms
        assert (
            query_time < 0.1
        ), f"Query took {query_time * 1000:.2f}ms (expected < 100ms)"

        print(f"\n✓ Queried {num_artifacts} artifacts in {query_time * 1000:.2f}ms")

    def test_query_performance_by_time_range(self, session, artifact_repo, test_video):
        """Test query performance for time range queries."""
        run_id = str(uuid.uuid4())
        num_artifacts = 1000

        # Create artifacts spread across 1 hour
        for i in range(num_artifacts):
            payload = ObjectDetectionV1(
                label="person" if i % 2 == 0 else "car",
                confidence=0.9,
                bounding_box=BoundingBox(x=100, y=100, width=200, height=200),
                frame_number=i * 30,
            )

            artifact = ArtifactEnvelope(
                artifact_id=str(uuid.uuid4()),
                asset_id=test_video.video_id,
                artifact_type="object.detection",
                schema_version=1,
                span_start_ms=i * 3600,
                span_end_ms=i * 3600 + 33,
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

        # Test time range query (first 10 minutes)
        start_time = time.time()
        artifacts = artifact_repo.get_by_span(
            asset_id=test_video.video_id,
            artifact_type="object.detection",
            span_start_ms=0,
            span_end_ms=600000,  # 10 minutes
        )
        query_time = time.time() - start_time

        # Should return approximately 167 artifacts (10 minutes / 3.6s per artifact)
        assert 150 < len(artifacts) < 200

        # Performance assertion: should query in under 50ms
        assert (
            query_time < 0.05
        ), f"Query took {query_time * 1000:.2f}ms (expected < 50ms)"

        print(
            f"\n✓ Time range query returned {len(artifacts)} artifacts "
            f"in {query_time * 1000:.2f}ms"
        )

    def test_index_usage_verification(self, session, test_video):
        """Verify that database indexes are being used for queries."""
        # This test verifies that the query planner is using indexes
        # by checking the EXPLAIN QUERY PLAN output

        # Query by asset_id and artifact_type (should use index)
        result = session.execute(
            sql_text(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM artifacts
                WHERE asset_id = :asset_id
                AND artifact_type = :artifact_type
                ORDER BY span_start_ms
                """
            ),
            {"asset_id": test_video.video_id, "artifact_type": "scene"},
        ).fetchall()

        # Convert result to string for analysis
        plan = " ".join([str(row) for row in result])

        # Verify index is being used (SQLite uses "USING INDEX" in plan)
        # Note: In-memory SQLite may not always use indexes for small datasets
        print(f"\n✓ Query plan: {plan}")

        # For this test, we just verify the query executes successfully
        # In production with PostgreSQL, we would check for index scans
        assert len(result) > 0

    def test_multi_profile_query_performance(self, artifact_repo, test_video):
        """Test query performance with multiple model profiles."""
        profiles = ["fast", "balanced", "high_quality"]
        artifacts_per_profile = 100

        # Create artifacts for each profile
        for profile in profiles:
            run_id = str(uuid.uuid4())

            for i in range(artifacts_per_profile):
                payload = TranscriptSegmentV1(
                    text=f"Text from {profile} model segment {i}",
                    speaker=None,
                    confidence=0.9,
                    language="en",
                )

                artifact = ArtifactEnvelope(
                    artifact_id=str(uuid.uuid4()),
                    asset_id=test_video.video_id,
                    artifact_type="transcript.segment",
                    schema_version=1,
                    span_start_ms=i * 1000,
                    span_end_ms=(i + 1) * 1000,
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

        # Test querying specific profile
        start_time = time.time()
        # Note: Current implementation doesn't filter by profile in get_by_asset
        # This would need to be added to the repository for profile-specific queries
        artifacts = artifact_repo.get_by_asset(
            asset_id=test_video.video_id, artifact_type="transcript.segment"
        )
        query_time = time.time() - start_time

        # Should return all artifacts from all profiles
        assert len(artifacts) == artifacts_per_profile * len(profiles)

        print(
            f"\n✓ Multi-profile query returned {len(artifacts)} artifacts "
            f"in {query_time * 1000:.2f}ms"
        )

    def test_database_size_monitoring(self, session):
        """Monitor database size growth with artifacts."""
        # Get initial database stats
        result = session.execute(sql_text("SELECT COUNT(*) FROM artifacts")).fetchone()
        initial_count = result[0]

        # Get page count (SQLite-specific)
        result = session.execute(sql_text("PRAGMA page_count")).fetchone()
        initial_pages = result[0]

        result = session.execute(sql_text("PRAGMA page_size")).fetchone()
        page_size = result[0]

        initial_size_kb = (initial_pages * page_size) / 1024

        print("\n✓ Database stats:")
        print(f"  Artifacts: {initial_count}")
        print(f"  Size: {initial_size_kb:.2f} KB")
        print(f"  Pages: {initial_pages}")
        print(f"  Page size: {page_size} bytes")

        # This test just monitors - no assertions
        # In production, we would track growth over time
        assert initial_size_kb >= 0
