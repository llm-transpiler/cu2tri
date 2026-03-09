from __future__ import annotations

import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/cu2tri"))
sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - optional dependency
    # Prefer NPU client implementation while keeping legacy symbol names.
    from server.npu.client import NPUClient as NVGPUClient  # type: ignore

    NVGPU_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        from server.nvgpu.client import NVGPUClient  # type: ignore

        NVGPU_AVAILABLE = True
    except Exception:
        NVGPU_AVAILABLE = False
        NVGPUClient = None  # type: ignore

__all__ = ["NVGPUClient", "NVGPU_AVAILABLE"]
