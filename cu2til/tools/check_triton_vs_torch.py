#!/usr/bin/env python3
"""
Convenience wrapper that validates Triton kernels against PyTorch references.
Supports optional multi-shape testing via get_all_cuda_torch_inputs.
"""
import argparse
import sys
from pathlib import Path

# Ensure testcase root and project root are importable
TESTCASE_ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTCASE_ROOT_DIR))

CASE_ROOT = Path.cwd()
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))

PROJECT_ROOT = None
for candidate in TESTCASE_ROOT_DIR.resolve().parents:
    if (candidate / "cu2til").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from get_data import (  # type: ignore
    Params,
    get_cuda_torch_inputs,
    cuda_output_tensor_transform,
)

try:
    from get_data import get_all_cuda_torch_inputs  # type: ignore
except ImportError:
    get_all_cuda_torch_inputs = None

from torch_.ref import torch_kernel  # type: ignore
from triton_.kernel import triton_kernel  # type: ignore
from cu2til.tools.checker import (
    check_triton_vs_torch,
    check_triton_vs_torch_dynamic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Triton kernel against PyTorch reference")
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Disable performance benchmarking (default: enabled)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use get_all_cuda_torch_inputs if available for multi-shape testing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    enable_perf = not args.no_perf
    params = Params()

    if args.dynamic and get_all_cuda_torch_inputs is not None:
        success = check_triton_vs_torch_dynamic(
            get_all_cuda_torch_inputs=get_all_cuda_torch_inputs,
            params=params,
            torch_kernel=torch_kernel,
            triton_kernel=triton_kernel,
            output_tensor_transform=cuda_output_tensor_transform,
            enable_perf=enable_perf,
        )
    else:
        if args.dynamic and get_all_cuda_torch_inputs is None:
            print("⚠️  --dynamic requested but get_all_cuda_torch_inputs is unavailable; falling back to single-shape mode.")
        success = check_triton_vs_torch(
            get_cuda_torch_inputs=get_cuda_torch_inputs,
            params=params,
            torch_kernel=torch_kernel,
            triton_kernel=triton_kernel,
            output_tensor_transform=cuda_output_tensor_transform,
            enable_perf=enable_perf,
        )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
