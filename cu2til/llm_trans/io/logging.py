from __future__ import annotations

import logging
from typing import Iterable

from ..config.settings import Settings


def configure_logger(settings: Settings) -> logging.Logger:
    logger = logging.getLogger("llm_trans")
    logger.setLevel(logging.DEBUG)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if settings.log_file is None:
        raise RuntimeError("Log file path has not been configured")

    file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    handlers: list[logging.Handler] = [file_handler]
    if settings.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        handlers.append(console_handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)-7s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


__all__ = ["configure_logger"]
