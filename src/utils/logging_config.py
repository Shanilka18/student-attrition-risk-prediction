"""Standardized structured logging configuration for the ERI pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.utils.paths import get_artifacts_dir


def setup_logger(
    name: str = "eri_pipeline",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    propagate: bool = False,
) -> logging.Logger:
    """Create and configure a standardized logger instance.

    Parameters
    ----------
    name : str
        Logger name identifier (usually module or component name).
    log_file : Optional[str]
        Optional file path (or relative to artifacts/logs) for log output.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    propagate : bool
        Whether to propagate records to root logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Clear existing handlers to prevent duplicate lines on re-initialization
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (stdout)
    # Ensure stdout reconfigured to utf-8 if possible, or fallback safely
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        file_path = Path(log_file)
        if not file_path.is_absolute():
            log_dir = get_artifacts_dir("logs")
            file_path = log_dir / log_file
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
