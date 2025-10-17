"""Timezone-aware date/time helpers shared across the project."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo
from typing import Any

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


def parse_timestamp(value: Any) -> datetime | None:
    """Parse a datetime-like value and normalize it to the default timezone.

    Accepts aware/naive ``datetime`` objects or ISO-8601 strings (with optional ``Z`` suffix).
    Returns ``None`` if the value cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_timezone(value)
    if isinstance(value, str):
        try:
            return ensure_timezone(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def normalize_timestamp_iso(value: Any) -> str | None:
    """Return an ISO-8601 string for ``value`` after timezone normalization.

    Falls back to the original string if parsing fails; otherwise returns ``None``.
    """
    parsed = parse_timestamp(value)
    if parsed is not None:
        return parsed.isoformat()
    if isinstance(value, str):
        return value
    return None

'''
>>> from utils.timezone import now_timestamp
>>> now_timestamp()
datetime.datetime(2025, 10, 17, 8, 24, 22, 223626, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))
>>> from datetime import datetime
>>> datetime.now()
datetime.datetime(2025, 10, 17, 0, 24, 58, 200549)
>>> datetime.now().isoformat()
'2025-10-17T00:25:20.139185'
>>> now_timestamp().isoformat()
'2025-10-17T08:25:24.448158+08:00'
>>> now_timestamp().strftime('%Y%m%d_%H%M%S')
'20251017_082537'
>>> datetime.now().strftime('%Y%m%d_%H%M%S')
'20251017_002543'
>>> from utils.timezone import format_timestamp
>>> format_timestamp()
'20251017_082715'
'''
