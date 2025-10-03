#!/usr/bin/env python3
"""Performance benchmark - measure GPU computation speed."""
import argparse
import sys
import time
import torch


def benchmark_matmul(size, iterations, warmup=10):
    """Benchmark matrix multiplication."""
    x = torch.randn(size, size, device="cuda")
    y = torch.randn(size, size, device="cuda")
    
    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate FLOPS
    flops = 2 * size ** 3 * iterations  # 2n^3 operations for matmul
    tflops = flops / elapsed / 1e12
    
    return elapsed, tflops


def benchmark_memory_bandwidth(size_mb, iterations):
    """Benchmark memory bandwidth."""
    elements = int(size_mb * 1e6 / 4)  # 4 bytes per float32
    
    # Host to Device
    host_tensor = torch.randn(elements)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        device_tensor = host_tensor.to("cuda")
    torch.cuda.synchronize()
    h2d_time = time.perf_counter() - start
    h2d_bandwidth = (size_mb * iterations) / h2d_time / 1e3  # GB/s
    
    # Device to Host
    start = time.perf_counter()
    for _ in range(iterations):
        host_result = device_tensor.to("cpu")
    torch.cuda.synchronize()
    d2h_time = time.perf_counter() - start
    d2h_bandwidth = (size_mb * iterations) / d2h_time / 1e3  # GB/s
    
    return h2d_bandwidth, d2h_bandwidth


def main():
    parser = argparse.ArgumentParser(description="GPU performance benchmark")
    parser.add_argument("--matmul-size", type=int, default=4096, help="Matrix size for matmul")
    parser.add_argument("--matmul-iters", type=int, default=100, help="Matmul iterations")
    parser.add_argument("--memory-size-mb", type=int, default=100, help="Memory transfer size in MB")
    parser.add_argument("--memory-iters", type=int, default=10, help="Memory transfer iterations")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory bandwidth test")
    args = parser.parse_args()
    
    print("=== GPU Performance Benchmark ===")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    
    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")
    print(f"Total Memory: {device_props.total_memory / 1e9:.2f} GB")
    print(f"Multi-Processors: {device_props.multi_processor_count}")
    print()
    
    try:
        # Matrix Multiplication Benchmark
        print(f"Matrix Multiplication ({args.matmul_size}x{args.matmul_size})...")
        elapsed, tflops = benchmark_matmul(args.matmul_size, args.matmul_iters)
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Average time: {elapsed/args.matmul_iters*1000:.2f}ms per iteration")
        print(f"  Performance: {tflops:.2f} TFLOPS")
        print()
        
        # Memory Bandwidth Benchmark
        if not args.skip_memory:
            print(f"Memory Bandwidth ({args.memory_size_mb}MB)...")
            h2d_bw, d2h_bw = benchmark_memory_bandwidth(args.memory_size_mb, args.memory_iters)
            print(f"  Host to Device: {h2d_bw:.2f} GB/s")
            print(f"  Device to Host: {d2h_bw:.2f} GB/s")
            print()
        
        print("=== BENCHMARK COMPLETED ===")
        return 0
        
    except Exception as e:
        print(f"BENCHMARK FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

