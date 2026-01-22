"""Tests for FindWithinVideoService."""

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.database.models import Video as VideoEntity
from src.domain.artifacts import ArtifactEnvelope
from src.domain.schema_initialization import register_all_schemas
from src.domain.schema_registry import SchemaRegistry
from src.repositories.artifact_repository import SqlArtifactRepository
from src.repositories.selection_policy_manager import SelectionPolicyManager
from src.services.find_within_video_service import FindWithinVideoService
from src.services.projection_sync_service import ProjectionSyncService


@pytest.fixture
def engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Create FTS5 tables for SQLite
    with engine.connect() as conn:
        # Create transcript FTS5 table
        conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE transcript_fts USING fts5(
                    artifact_id UNINDEXED,
                    asset_id UNINDEXED,
                    start_ms UNINDEXED,
                    end_ms UNINDEXED,
                    text
                )
                """
            )
        )

        # Create transcript metadata table
        conn.execute(
            text(
                """
                CREATE TABLE transcript_fts_metadata (
                    artifact_id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER NOT NULL
                )
                """
            )
        )

        # Create OCR FTS5 table
        conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE ocr_fts USING fts5(
                    artifact_id UNINDEXED,
                    asset_id UNINDEXED,
                    start_ms UNINDEXED,
                    end_ms UNINDEXED,
                    text
                )
                """
            )
        )

        # Create OCR metadata table
        conn.execute(
            text(
                """
                CREATE TABLE ocr_fts_metadata (
                    artifact_id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER NOT NULL
                )
                """
            )
        )

        conn.commit()

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
def artifact_repo(session, schema_registry):
    """Create artifact repository instance."""
    return SqlArtifactRepository(session, schema_registry)


@pytest.fixture
def policy_manager(session):
    """Create selection policy manager instance."""
    return SelectionPolicyManager(session)


@pytest.fixture
def projection_sync(session):
    """Create projection sync service instance."""
    return ProjectionSyncService(session)


@pytest.fixture
def find_service(session, policy_manager):
    """Create find within video service instance."""
    return FindWithinVideoService(session, policy_manager)


@pytest.fixture
def test_video(session):
    """Create a test video entity."""
    video = VideoEntity(
        video_id="test_video_1",
        file_path="/test/video.mp4",
        filename="video.mp4",
        last_modified=datetime.now(),
        status="completed",
    )
    session.add(video)
    session.commit()
    return video


def create_transcript_artifact(
    artifact_id, asset_id, start_ms, end_ms, text, confidence=0.9, run_id="run_1"
):
    """Helper to create transcript artifact."""
    payload = {"text": text, "confidence": confidence, "language": "en"}
    return ArtifactEnvelope(
        artifact_id=artifact_id,
        asset_id=asset_id,
        artifact_type="transcript.segment",
        schema_version=1,
        span_start_ms=start_ms,
        span_end_ms=end_ms,
        payload_json=json.dumps(payload),
        producer="whisper",
        producer_version="3.0.0",
        model_profile="balanced",
        config_hash="abc123",
        input_hash="def456",
        run_id=run_id,
        created_at=datetime.now(),
    )


def create_ocr_artifact(
    artifact_id, asset_id, start_ms, end_ms, text, confidence=0.9, run_id="run_1"
):
    """Helper to create OCR artifact."""
    payload = {
        "text": text,
        "confidence": confidence,
        "bounding_box": [
            {"x": 0.1, "y": 0.1},
            {"x": 0.9, "y": 0.1},
            {"x": 0.9, "y": 0.9},
            {"x": 0.1, "y": 0.9},
        ],
        "language": "en",
        "frame_number": start_ms // 33,  # Approximate frame number
    }
    return ArtifactEnvelope(
        artifact_id=artifact_id,
        asset_id=asset_id,
        artifact_type="ocr.text",
        schema_version=1,
        span_start_ms=start_ms,
        span_end_ms=end_ms,
        payload_json=json.dumps(payload),
        producer="easyocr",
        producer_version="1.0.0",
        model_profile="balanced",
        config_hash="abc123",
        input_hash="def456",
        run_id=run_id,
        created_at=datetime.now(),
    )


class TestFindNext:
    """Tests for find_next method."""

    def test_find_next_transcript_single_match(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test finding next transcript match with single result."""
        # Create transcript artifacts
        artifact1 = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "Hello world"
        )
        artifact2 = create_transcript_artifact(
            "art_2", test_video.video_id, 3000, 4000, "This is a test"
        )
        artifact3 = create_transcript_artifact(
            "art_3", test_video.video_id, 5000, 6000, "Another test here"
        )

        # Store artifacts and sync projections
        artifact_repo.create(artifact1)
        projection_sync.sync_artifact(artifact1)
        artifact_repo.create(artifact2)
        projection_sync.sync_artifact(artifact2)
        artifact_repo.create(artifact3)
        projection_sync.sync_artifact(artifact3)

        # Find next occurrence of "test" from 0ms
        results = find_service.find_next(test_video.video_id, "test", 0, "transcript")

        # Should find artifact2 first
        assert len(results) == 2
        assert results[0]["artifact_id"] == "art_2"
        assert results[0]["jump_to"]["start_ms"] == 3000
        assert results[0]["jump_to"]["end_ms"] == 4000
        assert results[0]["source"] == "transcript"
        assert "test" in results[0]["snippet"].lower()

    def test_find_next_transcript_from_middle(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test finding next transcript match starting from middle of video."""
        # Create transcript artifacts
        artifact1 = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "First test"
        )
        artifact2 = create_transcript_artifact(
            "art_2", test_video.video_id, 3000, 4000, "Second test"
        )
        artifact3 = create_transcript_artifact(
            "art_3", test_video.video_id, 5000, 6000, "Third test"
        )

        # Store artifacts and sync projections
        artifact_repo.create(artifact1)
        projection_sync.sync_artifact(artifact1)
        artifact_repo.create(artifact2)
        projection_sync.sync_artifact(artifact2)
        artifact_repo.create(artifact3)
        projection_sync.sync_artifact(artifact3)

        # Find next occurrence of "test" from 3500ms (middle of artifact2)
        results = find_service.find_next(
            test_video.video_id, "test", 3500, "transcript"
        )

        # Should find artifact3 (skipping artifact2 since we're past its start)
        assert len(results) == 1
        assert results[0]["artifact_id"] == "art_3"
        assert results[0]["jump_to"]["start_ms"] == 5000

    def test_find_next_ocr_single_match(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test finding next OCR match with single result."""
        # Create OCR artifacts
        artifact1 = create_ocr_artifact(
            "ocr_1", test_video.video_id, 1000, 1033, "Button text"
        )
        artifact2 = create_ocr_artifact(
            "ocr_2", test_video.video_id, 3000, 3033, "Click here"
        )
        artifact3 = create_ocr_artifact(
            "ocr_3", test_video.video_id, 5000, 5033, "Submit button"
        )

        # Store artifacts and sync projections
        artifact_repo.create(artifact1)
        projection_sync.sync_artifact(artifact1)
        artifact_repo.create(artifact2)
        projection_sync.sync_artifact(artifact2)
        artifact_repo.create(artifact3)
        projection_sync.sync_artifact(artifact3)

        # Find next occurrence of "button" from 0ms
        results = find_service.find_next(test_video.video_id, "button", 0, "ocr")

        # Should find artifact1 and artifact3
        assert len(results) == 2
        assert results[0]["artifact_id"] == "ocr_1"
        assert results[0]["source"] == "ocr"
        assert "button" in results[0]["snippet"].lower()

    def test_find_next_multi_source(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test finding next match across both transcript and OCR."""
        # Create transcript artifact
        transcript = create_transcript_artifact(
            "art_1", test_video.video_id, 2000, 3000, "Password reset"
        )
        # Create OCR artifact
        ocr = create_ocr_artifact(
            "ocr_1", test_video.video_id, 4000, 4033, "Reset button"
        )

        # Store artifacts and sync projections
        artifact_repo.create(transcript)
        projection_sync.sync_artifact(transcript)
        artifact_repo.create(ocr)
        projection_sync.sync_artifact(ocr)

        # Find next occurrence of "reset" from 0ms in all sources
        results = find_service.find_next(test_video.video_id, "reset", 0, "all")

        # Should find both, ordered by timestamp
        assert len(results) == 2
        assert results[0]["artifact_id"] == "art_1"
        assert results[0]["source"] == "transcript"
        assert results[1]["artifact_id"] == "ocr_1"
        assert results[1]["source"] == "ocr"

    def test_find_next_no_matches(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test finding next when no matches exist."""
        # Create artifact without the search term
        artifact = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "Hello world"
        )
        artifact_repo.create(artifact)
        projection_sync.sync_artifact(artifact)

        # Search for non-existent term
        results = find_service.find_next(
            test_video.video_id, "nonexistent", 0, "transcript"
        )

        assert len(results) == 0

    def test_find_next_ordering(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test that find_next returns results in ascending timestamp order."""
        # Create artifacts out of order
        artifact3 = create_transcript_artifact(
            "art_3", test_video.video_id, 5000, 6000, "test three"
        )
        artifact1 = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "test one"
        )
        artifact2 = create_transcript_artifact(
            "art_2", test_video.video_id, 3000, 4000, "test two"
        )

        # Store in random order
        artifact_repo.create(artifact3)
        projection_sync.sync_artifact(artifact3)
        artifact_repo.create(artifact1)
        projection_sync.sync_artifact(artifact1)
        artifact_repo.create(artifact2)
        projection_sync.sync_artifact(artifact2)

        # Find all occurrences
        results = find_service.find_next(test_video.video_id, "test", 0, "transcript")

        # Should be ordered by timestamp ascending
        assert len(results) == 3
        assert results[0]["jump_to"]["start_ms"] == 1000
        assert results[1]["jump_to"]["start_ms"] == 3000
        assert results[2]["jump_to"]["start_ms"] == 5000


class TestFindPrev:
    """Tests for find_prev method."""

    def test_find_prev_transcript_single_match(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test finding previous transcript match with single result."""
        # Create transcript artifacts
        artifact1 = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "First test"
        )
        artifact2 = create_transcript_artifact(
            "art_2", test_video.video_id, 3000, 4000, "Second test"
        )
        artifact3 = create_transcript_artifact(
            "art_3", test_video.video_id, 5000, 6000, "Third test"
        )

        # Store artifacts and sync projections
        artifact_repo.create(artifact1)
        projection_sync.sync_artifact(artifact1)
        artifact_repo.create(artifact2)
        projection_sync.sync_artifact(artifact2)
        artifact_repo.create(artifact3)
        projection_sync.sync_artifact(artifact3)

        # Find previous occurrence of "test" from 6000ms (end of video)
        results = find_service.find_prev(
            test_video.video_id, "test", 6000, "transcript"
        )

        # Should find artifact3 first (most recent before 6000ms)
        assert len(results) == 3
        assert results[0]["artifact_id"] == "art_3"
        assert results[0]["jump_to"]["start_ms"] == 5000

    def test_find_prev_from_middle(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test finding previous match starting from middle of video."""
        # Create transcript artifacts
        artifact1 = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "First test"
        )
        artifact2 = create_transcript_artifact(
            "art_2", test_video.video_id, 3000, 4000, "Second test"
        )
        artifact3 = create_transcript_artifact(
            "art_3", test_video.video_id, 5000, 6000, "Third test"
        )

        # Store artifacts and sync projections
        artifact_repo.create(artifact1)
        projection_sync.sync_artifact(artifact1)
        artifact_repo.create(artifact2)
        projection_sync.sync_artifact(artifact2)
        artifact_repo.create(artifact3)
        projection_sync.sync_artifact(artifact3)

        # Find previous occurrence of "test" from 3500ms
        results = find_service.find_prev(
            test_video.video_id, "test", 3500, "transcript"
        )

        # Should find artifact2 and artifact1 (before 3500ms)
        assert len(results) == 2
        assert results[0]["artifact_id"] == "art_2"
        assert results[1]["artifact_id"] == "art_1"

    def test_find_prev_ordering(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test that find_prev returns results in descending timestamp order."""
        # Create artifacts
        artifact1 = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "test one"
        )
        artifact2 = create_transcript_artifact(
            "art_2", test_video.video_id, 3000, 4000, "test two"
        )
        artifact3 = create_transcript_artifact(
            "art_3", test_video.video_id, 5000, 6000, "test three"
        )

        # Store artifacts
        artifact_repo.create(artifact1)
        projection_sync.sync_artifact(artifact1)
        artifact_repo.create(artifact2)
        projection_sync.sync_artifact(artifact2)
        artifact_repo.create(artifact3)
        projection_sync.sync_artifact(artifact3)

        # Find all occurrences from end
        results = find_service.find_prev(
            test_video.video_id, "test", 10000, "transcript"
        )

        # Should be ordered by timestamp descending
        assert len(results) == 3
        assert results[0]["jump_to"]["start_ms"] == 5000
        assert results[1]["jump_to"]["start_ms"] == 3000
        assert results[2]["jump_to"]["start_ms"] == 1000

    def test_find_prev_multi_source(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test finding previous match across both transcript and OCR."""
        # Create transcript artifact
        transcript = create_transcript_artifact(
            "art_1", test_video.video_id, 2000, 3000, "Password reset"
        )
        # Create OCR artifact
        ocr = create_ocr_artifact(
            "ocr_1", test_video.video_id, 4000, 4033, "Reset button"
        )

        # Store artifacts and sync projections
        artifact_repo.create(transcript)
        projection_sync.sync_artifact(transcript)
        artifact_repo.create(ocr)
        projection_sync.sync_artifact(ocr)

        # Find previous occurrence of "reset" from 10000ms in all sources
        results = find_service.find_prev(test_video.video_id, "reset", 10000, "all")

        # Should find both, ordered by timestamp descending
        assert len(results) == 2
        assert results[0]["artifact_id"] == "ocr_1"
        assert results[0]["source"] == "ocr"
        assert results[1]["artifact_id"] == "art_1"
        assert results[1]["source"] == "transcript"


class TestMultiSourceMerging:
    """Tests for multi-source search merging."""

    def test_multi_source_timestamp_ordering(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test that multi-source results are properly merged by timestamp."""
        # Create interleaved transcript and OCR artifacts
        transcript1 = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "search term"
        )
        ocr1 = create_ocr_artifact(
            "ocr_1", test_video.video_id, 1500, 1533, "search term"
        )
        transcript2 = create_transcript_artifact(
            "art_2", test_video.video_id, 3000, 4000, "search term"
        )
        ocr2 = create_ocr_artifact(
            "ocr_2", test_video.video_id, 3500, 3533, "search term"
        )

        # Store all artifacts
        for artifact in [transcript1, ocr1, transcript2, ocr2]:
            artifact_repo.create(artifact)
            projection_sync.sync_artifact(artifact)

        # Find all occurrences
        results = find_service.find_next(test_video.video_id, "search", 0, "all")

        # Should be properly interleaved by timestamp
        assert len(results) == 4
        assert results[0]["artifact_id"] == "art_1"
        assert results[0]["source"] == "transcript"
        assert results[1]["artifact_id"] == "ocr_1"
        assert results[1]["source"] == "ocr"
        assert results[2]["artifact_id"] == "art_2"
        assert results[2]["source"] == "transcript"
        assert results[3]["artifact_id"] == "ocr_2"
        assert results[3]["source"] == "ocr"

    def test_multi_source_with_source_tag(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test that each result is tagged with its source."""
        # Create one of each type
        transcript = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "keyword"
        )
        ocr = create_ocr_artifact("ocr_1", test_video.video_id, 2000, 2033, "keyword")

        # Store artifacts
        artifact_repo.create(transcript)
        projection_sync.sync_artifact(transcript)
        artifact_repo.create(ocr)
        projection_sync.sync_artifact(ocr)

        # Find all occurrences
        results = find_service.find_next(test_video.video_id, "keyword", 0, "all")

        # Verify source tags
        assert len(results) == 2
        assert results[0]["source"] == "transcript"
        assert results[1]["source"] == "ocr"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_query(self, find_service, test_video):
        """Test handling of empty query string."""
        # Empty query should return no results
        results = find_service.find_next(test_video.video_id, "", 0, "all")
        assert len(results) == 0

    def test_special_characters_in_query(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test handling of special characters in search query."""
        # Create artifact with special characters
        artifact = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "email@example.com"
        )
        artifact_repo.create(artifact)
        projection_sync.sync_artifact(artifact)

        # Search for email (FTS5 should handle this)
        results = find_service.find_next(test_video.video_id, "email", 0, "transcript")

        # Should find the match
        assert len(results) == 1

    def test_case_insensitive_search(
        self, find_service, artifact_repo, projection_sync, test_video
    ):
        """Test that search is case-insensitive."""
        # Create artifact with mixed case
        artifact = create_transcript_artifact(
            "art_1", test_video.video_id, 1000, 2000, "Hello World"
        )
        artifact_repo.create(artifact)
        projection_sync.sync_artifact(artifact)

        # Search with different case
        results = find_service.find_next(test_video.video_id, "hello", 0, "transcript")

        # Should find the match
        assert len(results) == 1
        assert "hello" in results[0]["snippet"].lower()

    def test_nonexistent_video(self, find_service):
        """Test searching in non-existent video."""
        # Search in video that doesn't exist
        results = find_service.find_next("nonexistent_video", "test", 0, "all")

        # Should return empty results
        assert len(results) == 0
