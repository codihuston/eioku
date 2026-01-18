"""Tests for file hash service."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.file_hash_service import FileHashError, FileHashService


class TestFileHashService:
    """Test FileHashService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = FileHashService()

    def test_calculate_hash_success(self):
        """Test successful hash calculation."""
        # Create temporary file with known content
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name

        try:
            # Calculate hash
            hash_value = self.service.calculate_hash(temp_path)

            # Verify hash is returned
            assert isinstance(hash_value, str)
            assert len(hash_value) == 16  # xxhash64 hex length

            # Hash should be consistent
            hash_value2 = self.service.calculate_hash(temp_path)
            assert hash_value == hash_value2

        finally:
            Path(temp_path).unlink()

    def test_calculate_hash_file_not_found(self):
        """Test hash calculation with missing file."""
        with pytest.raises(FileHashError, match="File not found"):
            self.service.calculate_hash("/nonexistent/file.txt")

    def test_calculate_hash_directory(self):
        """Test hash calculation with directory instead of file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileHashError, match="Path is not a file"):
                self.service.calculate_hash(temp_dir)

    def test_calculate_hash_permission_error(self):
        """Test hash calculation with permission error."""
        # Create a real file first, then mock the open to fail
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            with patch("builtins.open", side_effect=PermissionError("Access denied")):
                with pytest.raises(FileHashError, match="Failed to read file"):
                    self.service.calculate_hash(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_verify_hash_success(self):
        """Test successful hash verification."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            # Calculate hash
            expected_hash = self.service.calculate_hash(temp_path)

            # Verify hash matches
            assert self.service.verify_hash(temp_path, expected_hash) is True

            # Verify wrong hash doesn't match
            assert self.service.verify_hash(temp_path, "wrong_hash") is False

        finally:
            Path(temp_path).unlink()

    def test_verify_hash_file_error(self):
        """Test hash verification with file error."""
        # Should return False for missing file
        result = self.service.verify_hash("/nonexistent/file.txt", "some_hash")
        assert result is False

    def test_different_files_different_hashes(self):
        """Test that different files produce different hashes."""
        # Create two different temporary files
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f1:
            f1.write("Content 1")
            temp_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f2:
            f2.write("Content 2")
            temp_path2 = f2.name

        try:
            hash1 = self.service.calculate_hash(temp_path1)
            hash2 = self.service.calculate_hash(temp_path2)

            # Hashes should be different
            assert hash1 != hash2

        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()

    def test_large_file_chunked_reading(self):
        """Test hash calculation with large file using chunked reading."""
        # Create service with small chunk size
        service = FileHashService(chunk_size=4)

        # Create file larger than chunk size
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("This is a longer content that exceeds chunk size")
            temp_path = f.name

        try:
            # Should successfully calculate hash
            hash_value = service.calculate_hash(temp_path)
            assert isinstance(hash_value, str)
            assert len(hash_value) == 16

        finally:
            Path(temp_path).unlink()

    def test_empty_file(self):
        """Test hash calculation for empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name  # Empty file

        try:
            hash_value = self.service.calculate_hash(temp_path)
            assert isinstance(hash_value, str)
            assert len(hash_value) == 16

        finally:
            Path(temp_path).unlink()
