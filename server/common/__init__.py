"""
Common utilities shared across cu2tri components.

This module provides shared functionality for:
- Timezone-aware datetime handling
- Task reference formatting
- Context management
- Logging configuration
- Performance timing
"""

from .timezone import (
    now_timestamp,
    format_timestamp,
    parse_timestamp,
    normalize_timestamp_iso,
    set_default_timezone,
    get_timezone,
    UTC_TZ,
    SHANGHAI_TZ,
)
from .task_refs import format_task_ref
from .context import ManagedContext
from .logger import (
    get_logger,
    configure_logger,
    get_llm_trans_logger,
    get_cu2tri_logger,
    NumberedRotatingFileHandler,
    create_numbered_rotating_handler,
)
from .timer import (
    Timer,
    TimerSample,
    HostTimer,
    monotonic_elapsed_ms,
    monotonic_timestamp_ns,
)

__all__ = [
    # Timezone
    "now_timestamp",
    "format_timestamp",
    "parse_timestamp",
    "normalize_timestamp_iso",
    "set_default_timezone",
    "get_timezone",
    "UTC_TZ",
    "SHANGHAI_TZ",
    # Task refs
    "format_task_ref",
    # Context
    "ManagedContext",
    # Logger
    "get_logger",
    "configure_logger",
    "get_llm_trans_logger",
    "get_cu2tri_logger",
    "NumberedRotatingFileHandler",
    "create_numbered_rotating_handler",
    # Timer
    "Timer",
    "TimerSample",
    "HostTimer",
    "monotonic_elapsed_ms",
    "monotonic_timestamp_ns",
]
