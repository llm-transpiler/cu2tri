from __future__ import annotations

import logging
from pathlib import Path

from server.common.logger import get_logger
from ..config.settings import Settings


def configure_logger(settings: Settings) -> logging.Logger:
    """
    配置llm_trans专用的日志记录器

    Args:
        settings: Settings对象，包含日志配置信息

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    if settings.log_file is None:
        raise RuntimeError("Log file path has not been configured")

    # 使用通用的日志模块，传入特定的格式和配置
    custom_format = "%(asctime)s - %(levelname)-7s - %(message)s"

    # 默认行为：单文件追加写，不轮转也不截断（max_bytes=0, backup_count=0）
    logger = get_logger(
        logger_name="llm_trans",
        log_file=Path(settings.log_file),
        log_level="DEBUG",
        console_output=settings.console_output,
        log_format=custom_format,
        max_bytes=0,     # 0 表示永不触发轮转
        backup_count=0,  # 0 表示不保留备份文件（但因为不轮转，也就不会截断）
    )

    return logger


__all__ = ["configure_logger"]
