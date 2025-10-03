#!/usr/bin/env python3
"""Test script to verify GPU mapping configuration."""
import os
import sys

def main():
    print("=== GPU Mapping Test ===")
    print()
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print()
    
    # Try to import torch
    try:
        import torch
        
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print()
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                print(f"CUDA Device {i}:")
                print(f"  Name: {device_name}")
                print(f"  Compute Capability: {device_props.major}.{device_props.minor}")
                print(f"  Total Memory: {device_props.total_memory / 1e9:.2f} GB")
                print()
            
            # Test computation
            print("Testing computation on assigned GPU...")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print("✓ Computation successful")
            
            # Check memory usage
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"Memory allocated: {allocated:.3f} GB")
        else:
            print("WARNING: CUDA not available")
            return 1
        
        print()
        print("=== TEST PASSED ===")
        return 0
        
    except ImportError:
        print("WARNING: PyTorch not installed, using basic test")
        print()
        
        # Basic test without torch
        if cuda_visible != "not set":
            print(f"✓ CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")
            print("=== TEST PASSED ===")
            return 0
        else:
            print("✗ CUDA_VISIBLE_DEVICES not set")
            return 1


if __name__ == "__main__":
    sys.exit(main())

