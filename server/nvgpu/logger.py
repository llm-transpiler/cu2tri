"""Unified logger configuration for NVGPU server."""
import logging
import sys
from datetime import datetime
from pathlib import Path

from utils.timezone import get_timezone


class TimezoneFormatter(logging.Formatter):
    """Logging formatter that respects configured timezone."""

    def converter(self, timestamp: float):
        return datetime.fromtimestamp(timestamp, get_timezone()).timetuple()


def setup_logger(name: str = "nvgpu_server", log_file: str = None, history_log_file: str = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Setup a unified logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Current session log file (timestamped)
        history_log_file: History log file (appended across sessions)
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter with aligned log levels (8 chars wide, left-aligned)
    formatter = TimezoneFormatter(
        '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Current session file handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # History file handler (optional, append mode)
    if history_log_file:
        history_path = Path(history_log_file)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_handler = logging.FileHandler(history_log_file, mode='a')
        history_handler.setLevel(level)
        history_handler.setFormatter(formatter)
        logger.addHandler(history_handler)
    
    return logger

