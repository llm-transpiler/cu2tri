"""Timezone-aware date/time helpers shared across the project."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

# Default timezone can be overridden at runtime via ``set_default_timezone``.
DEFAULT_TZ_NAME = "Asia/Shanghai"


@lru_cache(maxsize=None)
def _load_timezone(name: str) -> ZoneInfo:
    """Load and cache timezone objects."""
    return ZoneInfo(name)


def set_default_timezone(name: str) -> None:
    """Update the default timezone used by helper functions."""
    global DEFAULT_TZ_NAME
    DEFAULT_TZ_NAME = name
    _load_timezone.cache_clear()


def get_timezone(name: str | None = None) -> ZoneInfo:
    """Get a timezone by name, defaulting to the configured default."""
    target = name or DEFAULT_TZ_NAME
    try:
        return _load_timezone(target)
    except Exception:
        return _load_timezone("UTC")


def now_timestamp(tz_name: str | None = None) -> datetime:
    """Return current time in the specified or default timezone."""
    return datetime.now(get_timezone(tz_name))


def ensure_timezone(dt: datetime, tz_name: str | None = None) -> datetime:
    """Convert naive or aware datetime to the specified/default timezone."""
    tz = get_timezone(tz_name)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def format_timestamp(
    dt: datetime | None = None,
    fmt: str = "%Y%m%d_%H%M%S",
    tz_name: str | None = None,
) -> str:
    """Format datetime using the specified/default timezone."""
    target = ensure_timezone(dt, tz_name) if dt else now_timestamp(tz_name)
    return target.strftime(fmt)
