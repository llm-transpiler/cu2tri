from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

from ..core.runtime import RuntimeContext


async def write_jsonl_log(context: RuntimeContext, log_entry: Mapping[str, Any] | Any) -> None:
    if context.settings.jsonl_file is None:
        raise RuntimeError("JSONL log file path is not configured")

    if dataclasses.is_dataclass(log_entry):
        payload = dataclasses.asdict(log_entry)
    elif isinstance(log_entry, Mapping):
        payload = dict(log_entry)
    else:
        payload = log_entry

    async with context.jsonl_lock:
        try:
            with open(context.settings.jsonl_file, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover
            context.logger.warning(f"Failed to write JSONL log: {exc}")


__all__ = ["write_jsonl_log"]
