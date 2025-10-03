#!/usr/bin/env python3
"""Long running task - simulates extended computation."""
import argparse
import sys
import time
import torch


def main():
    parser = argparse.ArgumentParser(description="Long running GPU task")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--report-interval", type=int, default=10, help="Progress report interval")
    parser.add_argument("--work-size", type=int, default=2000, help="Work size per iteration")
    args = parser.parse_args()
    
    print(f"=== Long Running Task ({args.duration}s) ===")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Starting computation for {args.duration} seconds...")
    
    start_time = time.time()
    last_report = start_time
    iteration = 0
    
    try:
        while time.time() - start_time < args.duration:
            # Do some work
            x = torch.randn(args.work_size, args.work_size, device="cuda")
            y = torch.matmul(x, x.T)
            torch.cuda.synchronize()
            
            iteration += 1
            
            # Report progress
            current_time = time.time()
            if current_time - last_report >= args.report_interval:
                elapsed = current_time - start_time
                remaining = args.duration - elapsed
                progress = (elapsed / args.duration) * 100
                memory_used = torch.cuda.memory_allocated(0) / 1e9
                
                print(f"[{elapsed:.0f}s] Progress: {progress:.1f}% | "
                      f"Iterations: {iteration} | Memory: {memory_used:.2f}GB | "
                      f"Remaining: {remaining:.0f}s")
                
                last_report = current_time
            
            # Clean up
            del x, y
        
        total_time = time.time() - start_time
        print(f"\n=== COMPLETED ===")
        print(f"Total iterations: {iteration}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average: {total_time/iteration*1000:.2f}ms per iteration")
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

