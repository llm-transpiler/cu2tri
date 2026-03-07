from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Iterable
import argparse

from ..clients import ModelClients
from ..config.settings import Settings


@dataclass
class RuntimeContext:
    settings: Settings
    model: ModelClients
    logger: logging.Logger
    available_cases: dict[str, Iterable[str]]
    jsonl_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    retry_log_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    nvgpu_available: bool = False

    @property
    def args(self) -> argparse.Namespace:
        return self.settings.args


__all__ = ["RuntimeContext"]
