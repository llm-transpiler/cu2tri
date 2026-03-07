#!/usr/bin/env python3
"""CLI entry point for NVGPU server.

This module provides the command-line interface for the nvgpu-server command.
"""
import sys
from pathlib import Path
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Manually load cu2tri package if needed (workaround for Python 3.12)
cu2tri_spec = importlib.util.spec_from_file_location("cu2tri", str(PROJECT_ROOT / "__init__.py"))
if cu2tri_spec and cu2tri_spec.loader:
    try:
        cu2tri_module = importlib.util.module_from_spec(cu2tri_spec)
        if 'cu2tri' not in sys.modules:
            sys.modules['cu2tri'] = cu2tri_module
            cu2tri_spec.loader.exec_module(cu2tri_module)
    except:
        pass  # Already loaded or error, continue

# Import main function for CLI entry point
from server.nvgpu.main import main as _main


def cli_main():
    """Entry point for nvgpu-server command."""
    _main()


if __name__ == "__main__":
    cli_main()
