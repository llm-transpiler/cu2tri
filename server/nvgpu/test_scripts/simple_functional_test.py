#!/usr/bin/env python3
"""Simple functional test - quick GPU availability check."""
import sys
import torch


def main():
    print("=== Simple Functional Test ===")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Simple computation
    try:
        x = torch.randn(100, 100, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print("Matrix multiplication: PASSED")
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

