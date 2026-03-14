"""
Structured Logging for Smart Traffic Management System
Provides colourful, timestamped logging with file + console output.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class _ColourFormatter(logging.Formatter):
    """Console formatter with ANSI colours."""

    COLOURS = {
        logging.DEBUG: "\033[36m",      # cyan
        logging.INFO: "\033[32m",       # green
        logging.WARNING: "\033[33m",    # yellow
        logging.ERROR: "\033[31m",      # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelno, self.RESET)
        record.levelname = f"{colour}{record.levelname:<8}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "traffic",
    log_dir: Optional[str] = "logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Logger name.
        log_dir: Directory for log files (None to skip file logging).
        level: Logging level.
        log_to_file: Whether to also log to a file.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ch.setFormatter(_ColourFormatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{ts}.log"), encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)

    return logger
