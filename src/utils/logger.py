"""Standardized logger initialization for reproducibility and traceability."""

import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = "PRI_Framework", level: int = logging.INFO) -> logging.Logger:
    """Initialize and return a structured logger instance.

    Args:
        name (Optional[str]): Name of the logger domain. Defaults to 'PRI_Framework'.
        level (int): Logging priority level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger with console stream formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
