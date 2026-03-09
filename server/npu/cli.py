#!/usr/bin/env python3
"""CLI entry point for NPU server."""
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cu2tri_spec = importlib.util.spec_from_file_location("cu2tri", str(PROJECT_ROOT / "__init__.py"))
if cu2tri_spec and cu2tri_spec.loader:
    try:
        cu2tri_module = importlib.util.module_from_spec(cu2tri_spec)
        if "cu2tri" not in sys.modules:
            sys.modules["cu2tri"] = cu2tri_module
            cu2tri_spec.loader.exec_module(cu2tri_module)
    except Exception:
        pass

from server.npu.main import main as _main


def cli_main():
    """Entry point for npu-server command."""
    _main()


if __name__ == "__main__":
    cli_main()
