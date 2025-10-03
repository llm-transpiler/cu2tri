#!/usr/bin/env python3
"""Memory stress test - allocate large tensors to test memory handling."""
import argparse
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description="GPU memory stress test")
    parser.add_argument("--size", type=int, default=10000, help="Matrix size")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--allocate-gb", type=float, default=None, help="GB to allocate")
    args = parser.parse_args()
    
    print("=== Memory Stress Test ===")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    
    device_props = torch.cuda.get_device_properties(0)
    total_memory_gb = device_props.total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    
    try:
        if args.allocate_gb:
            # Allocate specific amount of memory
            print(f"\nAllocating {args.allocate_gb:.2f} GB...")
            elements = int(args.allocate_gb * 1e9 / 4)  # 4 bytes per float32
            tensor = torch.randn(elements, device="cuda")
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"Allocated: {allocated:.2f} GB")
            
            # Do some work
            result = tensor.sum()
            torch.cuda.synchronize()
            print(f"Sum computation: PASSED")
            
        else:
            # Run matrix operations
            print(f"\nRunning {args.iterations} iterations with {args.size}x{args.size} matrices...")
            for i in range(args.iterations):
                x = torch.randn(args.size, args.size, device="cuda")
                y = x @ x.T
                torch.cuda.synchronize()
                
                allocated = torch.cuda.memory_allocated(0) / 1e9
                print(f"Iteration {i+1}/{args.iterations}: {allocated:.2f} GB allocated")
                
                # Clean up
                del x, y
                torch.cuda.empty_cache()
        
        print("\n=== ALL TESTS PASSED ===")
        return 0
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OUT OF MEMORY: {e}", file=sys.stderr)
        else:
            print(f"RUNTIME ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

