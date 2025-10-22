from __future__ import annotations

import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
sys.path.insert(0, PROJECT_ROOT / "server" / "nvgpu")

try:  # pragma: no cover - optional dependency
    from server.nvgpu.client import NVGPUClient  # type: ignore

    NVGPU_AVAILABLE = True
except Exception:  # pragma: no cover
    NVGPU_AVAILABLE = False
    NVGPUClient = None  # type: ignore

__all__ = ["NVGPUClient", "NVGPU_AVAILABLE"]
