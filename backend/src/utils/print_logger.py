"""Simple print-based logger for development."""

import logging
import sys


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance with print-style formatting."""
    logger = logging.getLogger(name or __name__)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
