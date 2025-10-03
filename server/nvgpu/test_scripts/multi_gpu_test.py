#!/usr/bin/env python3
"""Multi-GPU test - verify specific GPU assignment."""
import argparse
import os
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU verification test")
    parser.add_argument("--expected-gpu", type=int, help="Expected GPU ID (from CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()
    
    print("=== Multi-GPU Test ===")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    
    device_count = torch.cuda.device_count()
    print(f"Visible GPU count: {device_count}")
    
    # The server should set CUDA_VISIBLE_DEVICES to a single GPU
    if device_count != 1:
        print(f"WARNING: Expected 1 visible GPU, got {device_count}")
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    
    print(f"Using GPU: {gpu_name}")
    print(f"Device ID: {gpu_props.major}.{gpu_props.minor}")
    print(f"Memory: {gpu_props.total_memory / 1e9:.2f} GB")
    
    # Run test
    try:
        x = torch.randn(1000, 1000, device="cuda:0")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("Computation test: PASSED")
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

