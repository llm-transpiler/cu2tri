#!/usr/bin/env python3
"""Example test script for NVGPU server.

This demonstrates the format that tasks should follow.
"""
import argparse
import sys
import time
import torch


def main():
    parser = argparse.ArgumentParser(description="Example GPU test script")
    parser.add_argument("--no-perf", action="store_true", help="Skip performance test")
    parser.add_argument("--sleep", type=int, default=5, help="Sleep duration in seconds")
    args = parser.parse_args()
    
    print("Starting GPU test...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.2f} GB")
    
    # Basic functional check
    print("\n=== Functional Check ===")
    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        print("Matrix multiplication: OK")
    except Exception as e:
        print(f"Functional check FAILED: {e}", file=sys.stderr)
        return 1
    
    # Performance test (if not skipped)
    if not args.no_perf:
        print("\n=== Performance Test ===")
        start = time.time()
        for i in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"100 iterations: {elapsed:.2f}s")
        print(f"Average: {elapsed/100*1000:.2f}ms per iteration")
    
    # Simulate some work
    print(f"\nSleeping for {args.sleep} seconds...")
    time.sleep(args.sleep)
    
    print("\n=== PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

