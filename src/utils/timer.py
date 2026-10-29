"""Execution timing utilities for pipeline stages and profiling."""

import functools
import logging
import time
from typing import Any, Callable, Optional


class Timer:
    """Context manager for measuring and logging execution duration.

    Usage
    -----
    with Timer("Data generation"):
        generate_synthetic_data()
    """

    def __init__(self, name: str = "Code block", logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.logger = logger
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_seconds = self.end_time - self.start_time
        message = f"[{self.name}] completed in {self.elapsed_seconds:.4f}s"
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def timed(name: Optional[str] = None, logger: Optional[logging.Logger] = None) -> Callable:
    """Decorator to time and report execution duration of any callable function.

    Parameters
    ----------
    name : Optional[str]
        Human-readable label for the operation (defaults to function __name__).
    logger : Optional[logging.Logger]
        Logger to receive the timing report. If None, prints to stdout.
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(op_name, logger=logger):
                return func(*args, **kwargs)

        return wrapper
    return decorator
