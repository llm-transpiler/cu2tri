import unittest
import sys
import os
import glob
from pathlib import Path

# Ensure we can import NVGPU client
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from client import NVGPUClient


def server_available() -> bool:
    try:
        client = NVGPUClient("http://localhost:8080")
        return client.health_check()
    except Exception:
        return False


requires_server = unittest.skipUnless(
    server_available(),
    "NVGPU server is not running on http://localhost:8080"
)


def get_client() -> NVGPUClient:
    return NVGPUClient("http://localhost:8080")


def get_server_log_path() -> Path | None:
    """Return path to the active server log file.

    Priority:
    1) NVGPU_SERVER_LOG env var if set and exists
    2) Latest server/nvgpu/logs/nvgpu_server_*.log
    """
    env_path = os.environ.get("NVGPU_SERVER_LOG")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    candidates = sorted(glob.glob(str(logs_dir / "nvgpu_server_*.log")))
    if candidates:
        return Path(candidates[-1])
    return None


def get_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return 0


def read_new_log_content(path: Path, offset: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return f.read()
    except Exception:
        return ""


def assert_patterns_in_order(testcase: unittest.TestCase, text: str, patterns: list[str], msg: str | None = None):
    """Assert regex patterns appear in text in the specified order.

    - Each pattern is a regex string; we search for the first occurrence.
    - Fails if any pattern is missing or order is violated.
    """
    import re
    indices: list[int] = []
    for p in patterns:
        m = re.search(p, text, flags=re.S)
        testcase.assertIsNotNone(m, msg or f"pattern not found: {p}")
        indices.append(m.start())
    testcase.assertEqual(sorted(indices), indices, msg or "patterns not in order")
