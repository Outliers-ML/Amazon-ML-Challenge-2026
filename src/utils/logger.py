"""Structured logging utility for experiments and pipelines.

Configures both console handlers with formatted output and persistent file logging
in the specified logs directory.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "AmazonML", log_file: Optional[Path | str] = None, level: int = logging.INFO) -> logging.Logger:
    """Instantiate or retrieve a configured logger.

    Args:
        name: Name of the logger, typically __name__ or project module name.
        log_file: Optional path to a file where logs should be appended.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        A fully configured `logging.Logger` instance.

    Example:
        >>> logger = get_logger("TrainPipeline", log_file="logs/train.log")
        >>> logger.info("Model training initiated.")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
