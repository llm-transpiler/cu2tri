#!/usr/bin/env python3
"""
Generic test script for triton kernels with multiple input configurations
"""

import os
import sys
import argparse
import time
import torch
from typing import List, Tuple, Any, Callable, Dict

# Add current directory to path for imports
TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description='Generic Triton kernel test with multiple inputs')
    parser.add_argument('--no-perf', action='store_true',
                       help='Disable performance testing (default: False)')
    parser.add_argument('--gpu-index', type=int, default=None,
                       help='GPU device index to use')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                       help='Tolerance for numerical comparison')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()

def check_triton_vs_torch_generic(
    test_cases: List[Tuple[Any, ...]],
    triton_kernel: Callable,
    torch_kernel: Callable = None,
    tolerance: float = 1e-3,
    enable_perf: bool = True,
    warmup: int = 10,
    iters: int = 100,
    verbose: bool = False
) -> bool:
    """
    Generic test function for triton vs torch comparison

    Args:
        test_cases: List of input tuples
        triton_kernel: Function that runs triton kernel
        torch_kernel: Function that runs torch reference (optional)
        tolerance: Numerical tolerance for comparison
        enable_perf: Whether to run performance tests
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        verbose: Enable detailed output

    Returns:
        bool: True if all tests pass
    """
    all_passed = True

    for i, inputs in enumerate(test_cases):
        print(f"\n=== Test case {i+1}/{len(test_cases)} ===")

        # Print input shapes
        if isinstance(inputs, (list, tuple)):
            if verbose:
                for j, inp in enumerate(inputs):
                    if hasattr(inp, 'shape'):
                        print(f"Input {j}: {inp.shape}, dtype: {inp.dtype}")
        else:
            if hasattr(inputs, 'shape'):
                print(f"Input: {inputs.shape}, dtype: {inputs.dtype}")

        try:
            # Run triton kernel
            triton_result = triton_kernel(*inputs) if isinstance(inputs, (list, tuple)) else triton_kernel(inputs)

            # Run torch reference if available
            if torch_kernel is not None:
                torch_result = torch_kernel(*inputs) if isinstance(inputs, (list, tuple)) else torch_kernel(inputs)

                # Compare results
                if hasattr(triton_result, 'shape') and hasattr(torch_result, 'shape'):
                    if triton_result.shape != torch_result.shape:
                        print(f"❌ Shape mismatch: triton {triton_result.shape} vs torch {torch_result.shape}")
                        all_passed = False
                        continue

                    diff = (triton_result - torch_result).abs()
                    max_diff = diff.max().item()

                    if max_diff > tolerance:
                        print(f"❌ Numerical mismatch: max diff = {max_diff:.6f} (tolerance: {tolerance})")
                        if verbose:
                            print(f"Triton result range: [{triton_result.min().item():.6f}, {triton_result.max().item():.6f}]")
                            print(f"Torch result range: [{torch_result.min().item():.6f}, {torch_result.max().item():.6f}]")
                        all_passed = False
                    else:
                        print(f"✅ Correctness check passed (max diff: {max_diff:.6f})")
                else:
                    print("⚠️  Cannot compare results - invalid tensor shapes")

            # Performance test if enabled
            if enable_perf and hasattr(triton_result, 'shape'):
                # Warmup
                for _ in range(warmup):
                    _ = triton_kernel(*inputs) if isinstance(inputs, (list, tuple)) else triton_kernel(inputs)

                torch.cuda.synchronize()

                # Benchmark
                start_time = time.perf_counter()
                for _ in range(iters):
                    result = triton_kernel(*inputs) if isinstance(inputs, (list, tuple)) else triton_kernel(inputs)
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                avg_time = (end_time - start_time) / iters * 1000  # ms

                # Calculate throughput
                if hasattr(inputs, 'numel'):
                    total_elements = inputs.numel()
                elif isinstance(inputs, (list, tuple)):
                    total_elements = sum(getattr(inp, 'numel', 0) for inp in inputs)
                else:
                    total_elements = 0

                if total_elements > 0:
                    throughput = total_elements / (avg_time / 1000) / 1e9  # GB/s assuming 4 bytes per element
                    print(f"📊 Performance: {avg_time:.3f} ms, {throughput:.2f} GB/s")
                else:
                    print(f"📊 Performance: {avg_time:.3f} ms")

        except Exception as e:
            print(f"❌ Test case {i+1} failed with exception: {e}")
            all_passed = False
            continue

    return all_passed

def main():
    args = parse_args()

    # Set GPU device
    if args.gpu_index is not None:
        torch.cuda.set_device(args.gpu_index)
        print(f"Using GPU {args.gpu_index}")

    # Try to import test data and kernels
    try:
        from get_data import get_test_cases
        test_cases = get_test_cases()
    except ImportError:
        print("❌ get_data.py not found or doesn't contain get_test_cases function")
        return False

    try:
        from triton_.kernel import triton_kernel
    except ImportError:
        print("❌ triton_/kernel.py not found or doesn't contain triton_kernel function")
        return False

    # Try to import torch reference
    torch_kernel = None
    try:
        from torch_.ref import torch_kernel
    except ImportError:
        print("⚠️  torch_/ref.py not found, skipping torch reference comparison")

    # Run tests
    success = check_triton_vs_torch_generic(
        test_cases=test_cases,
        triton_kernel=triton_kernel,
        torch_kernel=torch_kernel,
        tolerance=args.tolerance,
        enable_perf=not args.no_perf,
        warmup=args.warmup,
        iters=args.iters,
        verbose=args.verbose
    )

    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)