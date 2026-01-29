"""Unit tests for input validation module."""

import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.utils.hashing import compute_input_hash
from src.utils.input_validation import validate_inference_input


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        # Write some test data
        f.write(b"fake video data for testing")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestValidateInferenceInput:
    """Tests for validate_inference_input function."""

    def test_valid_input_passes_validation(self, temp_video_file):
        """Test that valid input with correct hash passes validation."""
        # Compute correct hash
        correct_hash = compute_input_hash(temp_video_file)

        # Should not raise
        validate_inference_input(temp_video_file, correct_hash)

    def test_file_not_found_raises_400(self):
        """Test that missing file raises 400 error."""
        with pytest.raises(HTTPException) as exc_info:
            validate_inference_input("/nonexistent/path/video.mp4", "somehash")

        assert exc_info.value.status_code == 400
        assert "not found" in exc_info.value.detail.lower()

    def test_hash_mismatch_raises_400(self, temp_video_file):
        """Test that hash mismatch raises 400 error."""
        wrong_hash = "0000000000000000"  # Obviously wrong hash

        with pytest.raises(HTTPException) as exc_info:
            validate_inference_input(temp_video_file, wrong_hash)

        assert exc_info.value.status_code == 400
        assert "mismatch" in exc_info.value.detail.lower()

    def test_directory_instead_of_file_raises_400(self):
        """Test that passing a directory instead of file raises 400 error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(HTTPException) as exc_info:
                validate_inference_input(temp_dir, "somehash")

            assert exc_info.value.status_code == 400
            assert "not a file" in exc_info.value.detail.lower()

    def test_file_modification_detected(self, temp_video_file):
        """Test that file modification is detected."""
        # Get hash of original file
        original_hash = compute_input_hash(temp_video_file)

        # Modify the file
        with open(temp_video_file, "ab") as f:
            f.write(b"modified content")

        # Validation should fail
        with pytest.raises(HTTPException) as exc_info:
            validate_inference_input(temp_video_file, original_hash)

        assert exc_info.value.status_code == 400
        assert "mismatch" in exc_info.value.detail.lower()

    def test_hash_consistency(self, temp_video_file):
        """Test that hash is consistent across multiple calls."""
        hash1 = compute_input_hash(temp_video_file)
        hash2 = compute_input_hash(temp_video_file)

        assert hash1 == hash2

        # Both should pass validation
        validate_inference_input(temp_video_file, hash1)
        validate_inference_input(temp_video_file, hash2)

    def test_different_files_have_different_hashes(self):
        """Test that different files produce different hashes."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f1:
            f1.write(b"content1")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f2:
            f2.write(b"content2")
            path2 = f2.name

        try:
            hash1 = compute_input_hash(path1)
            hash2 = compute_input_hash(path2)

            assert hash1 != hash2

            # Cross-validation should fail
            with pytest.raises(HTTPException):
                validate_inference_input(path1, hash2)

            with pytest.raises(HTTPException):
                validate_inference_input(path2, hash1)
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)
