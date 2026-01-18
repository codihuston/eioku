"""File hash calculation service using xxhash."""

import logging
from pathlib import Path

import xxhash

logger = logging.getLogger(__name__)


class FileHashError(Exception):
    """Exception raised when file hashing fails."""

    pass


class FileHashService:
    """Service for calculating file hashes using xxhash."""

    def __init__(self, chunk_size: int = 8192):
        """Initialize hash service.

        Args:
            chunk_size: Size of chunks to read for large files
        """
        self.chunk_size = chunk_size

    def calculate_hash(self, file_path: str) -> str:
        """Calculate xxhash64 hash of a file.

        Args:
            file_path: Path to the file to hash

        Returns:
            Hexadecimal xxhash64 hash string

        Raises:
            FileHashError: If file cannot be read or hashed
        """
        try:
            path = Path(file_path)

            if not path.exists():
                raise FileHashError(f"File not found: {file_path}")

            if not path.is_file():
                raise FileHashError(f"Path is not a file: {file_path}")

            logger.info(f"Calculating xxhash64 for {file_path}")

            hasher = xxhash.xxh64()

            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                while chunk := f.read(self.chunk_size):
                    hasher.update(chunk)

            hash_value = hasher.hexdigest()

            logger.info(f"Hash calculated for {file_path}: {hash_value}")
            return hash_value

        except OSError as e:
            error_msg = f"Failed to read file {file_path}: {e}"
            logger.error(error_msg)
            raise FileHashError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error hashing file {file_path}: {e}"
            logger.error(error_msg)
            raise FileHashError(error_msg)

    def verify_hash(self, file_path: str, expected_hash: str) -> bool:
        """Verify if file matches expected hash.

        Args:
            file_path: Path to the file to verify
            expected_hash: Expected xxhash64 hash

        Returns:
            True if hash matches, False otherwise
        """
        try:
            actual_hash = self.calculate_hash(file_path)
            return actual_hash.lower() == expected_hash.lower()
        except FileHashError:
            return False
