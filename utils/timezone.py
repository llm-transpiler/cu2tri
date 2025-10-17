"""Timezone-aware date/time helpers shared across the project."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import os
import time
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
        return _load_timezone(DEFAULT_TZ_NAME)


# Common timezone constants for convenience
UTC_TZ = get_timezone("UTC")
SHANGHAI_TZ = get_timezone("Asia/Shanghai")


def now_timestamp(tz_name: str | None = None) -> datetime:
    """Return current time in the specified or default timezone."""
    return datetime.now(get_timezone(tz_name))


def ensure_timezone(dt: datetime, tz_name: str | None = None) -> datetime:
    """Convert naive or aware datetime to the specified/default timezone."""
    tz = get_timezone(tz_name)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def to_utc(dt: datetime) -> datetime:
    """Convert a datetime to UTC, preserving the point in time."""
    return ensure_timezone(dt, "UTC")


def to_shanghai(dt: datetime) -> datetime:
    """Convert a datetime to Shanghai, preserving the point in time."""
    return ensure_timezone(dt, "Asia/Shanghai")


def format_timestamp(
    dt: datetime | None = None,
    fmt: str = "%Y%m%d_%H%M%S",
    tz_name: str | None = None,
) -> str:
    """Format datetime using the specified/default timezone."""
    target = ensure_timezone(dt, tz_name) if dt else now_timestamp(tz_name)
    return target.strftime(fmt)


def parse_timestamp(value: datetime | str | None) -> datetime | None:
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


def normalize_timestamp_iso(value: datetime | str | None) -> str | None:
    """Return an ISO-8601 string for ``value`` after timezone normalization.

    Falls back to the original string if parsing fails; otherwise returns ``None``.
    """
    parsed = parse_timestamp(value)
    if parsed is not None:
        return parsed.isoformat()
    if isinstance(value, str):
        return value
    return None


def apply_default_timezone_to_os(tz_name: str | None = None) -> None:
    """Apply the specified/default timezone to the OS-level TZ setting."""
    tz = get_timezone(tz_name)
    os.environ['TZ'] = getattr(tz, 'key', DEFAULT_TZ_NAME)
    try:
        time.tzset()
    except AttributeError:
        pass


'''
# 获取带时区的当前时间（默认 Asia/Shanghai）
>>> from utils.timezone import now_timestamp, format_timestamp, normalize_timestamp_iso, parse_timestamp
>>> now_timestamp()
datetime.datetime(2025, 10, 17, 8, 24, 22, 223626, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))

# 系统默认的 datetime.now() 没有时区信息
>>> from datetime import datetime
>>> datetime.now()
datetime.datetime(2025, 10, 17, 0, 24, 58, 200549)

# datetime.isoformat() 展示当前时间（无时区时没有 +xx:xx 后缀）
>>> datetime.now().isoformat()
'2025-10-17T00:25:20.139185'
# now_timestamp() 默认有时区信息（+08:00）
>>> now_timestamp().isoformat()
'2025-10-17T08:25:24.448158+08:00'

# 时间字符串格式化（年月日_时分秒形式）
>>> now_timestamp().strftime('%Y%m%d_%H%M%S')
'20251017_082537'
>>> datetime.now().strftime('%Y%m%d_%H%M%S')
'20251017_002543'

# 快捷调用 format_timestamp 得到当前时间字符串
>>> from utils.timezone import format_timestamp
>>> format_timestamp()
'20251017_082715'

# normalize_timestamp_iso 可将 ISO 字符串标准化（自动转为 Asia/Shanghai 时区带时区后缀）
>>> normalize_timestamp_iso('2022-11-20T14:00:00Z')
'2022-11-20T22:00:00+08:00'   # 将 UTC 时间转为上海时间

# parse_timestamp 可将字符串解析为 datetime 对象，并设定为 Asia/Shanghai 时区
# parse_timestamp(datetime.now())          # datetime对象
# parse_timestamp('2022-11-20T14:00:00Z')  # ISO 8601带Z后缀
# parse_timestamp('2022-11-20T10:30:00')   # ISO 8601无时区
>>> parse_timestamp('2022-11-20T10:30:00')
datetime.datetime(2022, 11, 20, 10, 30, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))

# 特别注意：所有工具函数默认输出本地 Asia/Shanghai 时区。输入带 'Z'（即 UTC）/无时区字符串，都会自动转换。



# ❌ 错误：直接replace会改变时间含义
dt_naive = datetime(2025, 10, 17, 10, 0, 0)
dt_naive.replace(tzinfo=ZoneInfo('UTC'))  # 10:00 UTC（实际比原时间晚）

# ✅ 正确：ensure_timezone会根据情况处理
ensure_timezone(dt_naive)  # 10:00 Asia/Shanghai（保持原意）



# ❌ Python 3.10 及之前版本不支持 Z
datetime.fromisoformat("2022-11-20T14:00:00Z")
# ValueError: Invalid isoformat string: '2022-11-20T14:00:00Z'

# ✅ 只接受标准的偏移量格式
datetime.fromisoformat("2022-11-20T14:00:00+00:00")
# datetime.datetime(2022, 11, 20, 14, 0, tzinfo=datetime.timezone.utc)
'''
