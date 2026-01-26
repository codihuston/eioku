"""Logger adapter that uses standard Python logging with JSON formatting."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a standard Python logger instance.
    
    This function returns a standard logging.Logger that will use the JSON
    formatter configured in main_api.py and main_worker.py. All logs will be
    structured as JSON for better log aggregation and parsing.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger instance configured with JSON formatting
    """
    return logging.getLogger(name)
