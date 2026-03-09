from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/cu2tri"))
sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - optional dependency
    from server.npu.client import NPUClient  # type: ignore

    NPU_AVAILABLE = True
except Exception:  # pragma: no cover
    NPU_AVAILABLE = False
    NPUClient = None  # type: ignore

__all__ = ["NPUClient", "NPU_AVAILABLE"]

