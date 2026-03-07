from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Mapping

from server.common.timezone import ensure_timezone, now_timestamp

from ..core.runtime import RuntimeContext

_NETWORK_PATTERNS = [
    r"connection error",
    r"connection reset",
    r"peer closed",
    r"timeout",
    r"timed out",
    r"eof occurred",
    r"incomplete chunked read",
    r"broken pipe",
    r"connection refused",
    r"connection aborted",
    r"network is unreachable",
    r"no route to host",
    r"connection pool",
    r"read timed out",
    r"connect timeout",
]

_RETRYABLE_PATTERNS = [
    r"overloaded",
    r"rate limit",
    r"quota exceeded",
    r"too many requests",
    r"service unavailable",
    r"temporarily unavailable",
    r"try again later",
    r"error code: 503",
    r"error code: 429",
    r"error code: 502",
    r"status: unavailable",
    r"status: resource_exhausted",
    r"capacity",
    r"busy",
    r"throttled",
]


def is_network_error(error_message: str) -> bool:
    error_str = str(error_message).lower()
    return any(re.search(pattern, error_str) for pattern in _NETWORK_PATTERNS)


def is_retryable_error(error_message: str) -> bool:
    error_str = str(error_message).lower()
    if any(re.search(pattern, error_str) for pattern in _RETRYABLE_PATTERNS):
        return True
    return is_network_error(error_message)


def _serialize_for_logging(data: Any) -> Any:
    try:
        return json.loads(json.dumps(data, ensure_ascii=False, default=str))
    except Exception as exc:  # pragma: no cover
        return f"<<serialization_failed: {exc}>>"


def build_retry_context(
    *,
    conversation_history: Any | None = None,
    retry_count: int | None = None,
    max_retries: int | None = None,
    api_params: Mapping[str, Any] | None = None,
    additional_meta: Mapping[str, Any] | None = None,
) -> dict:
    context: dict[str, Any] = {}
    if retry_count is not None:
        context["retry_count"] = retry_count
    if max_retries is not None:
        context["max_retries"] = max_retries
    if conversation_history is not None:
        context["conversation"] = _serialize_for_logging(conversation_history)
        context["conversation_message_count"] = len(conversation_history)
    if api_params is not None:
        sanitized = {k: v for k, v in api_params.items() if k != "messages"}
        context["api_params"] = _serialize_for_logging(sanitized)
    if additional_meta:
        context["meta"] = _serialize_for_logging(additional_meta)
    return context


async def log_retry_event(
    context: RuntimeContext,
    test_work_dir: Path,
    *,
    event_type: str,
    stage: str,
    case_type: str,
    case_name: str,
    attempt_number: int,
    round_id: int,
    retry_index: int,
    extra: Mapping[str, Any] | None = None,
    context_meta: Mapping[str, Any] | None = None,
) -> None:
    retry_log_dir = test_work_dir / "logs"
    retry_log_dir.mkdir(parents=True, exist_ok=True)
    retry_log_file = retry_log_dir / "retry_events.jsonl"
    record = {
        "_type": "retry_event",
        "event": event_type,
        "stage": stage,
        "case_type": case_type,
        "case_name": case_name,
        "attempt_number": attempt_number,
        "round_id": round_id,
        "retry_index": retry_index,
        "timestamp": ensure_timezone(now_timestamp()).isoformat(),
    }
    if extra:
        record.update(_serialize_for_logging(extra))
    if context_meta is not None:
        record["context"] = _serialize_for_logging(context_meta)

    async with context.retry_log_lock:
        try:
            with open(retry_log_file, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover
            context.logger.warning(f"Failed to persist retry event: {exc}")


__all__ = [
    "is_network_error",
    "is_retryable_error",
    "build_retry_context",
    "log_retry_event",
]
