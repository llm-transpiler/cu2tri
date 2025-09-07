import torch
import numpy as np
import ctypes
import os
import time
import subprocess
import sys
from torch.nn import functional as F
from cu2til.tools.builder import compile_cuda_kernel, CUDA_FOLDER_NAME, SEED, load_cuda_kernel

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

import sys
sys.path.insert(0, TESTCASE_ROOT_DIR)

from torch_.ref import torch_kernel
from cu2til.tools.checker import compare_results

def get_inputs():
    """Create test data"""
    torch.manual_seed(SEED)
    # GEMM operation: gemm_1024_128_128
    # GEMM: A(1024,128) @ B(128,128) = C(1024,128)
    A = torch.randn(1024, 128, dtype=torch.float16, device="cuda")
    B = torch.randn(128, 128, dtype=torch.float16, device="cuda")
    return A, B

def run_performance_test(A, B, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, (A, B))
    
    """Call CUDA GEMM kernel - directly use GPU tensor"""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Matrix dimensions must match"
    
    # Create output tensor (float32 for GEMM)
    C = torch.empty(m, n, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (A_ptr, B_ptr, C_ptr, m, k, n))
    
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 GEMM CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename)
    m, k, n = 1024, 128, 128
    print(f"📊 Test parameters: GEMM A({m},{k}) @ B({k},{n}) = C({m},{n})")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # A (half* GPU pointer)
            ctypes.c_void_p,  # B (half* GPU pointer)  
            ctypes.c_void_p,  # C (float* GPU pointer)
            ctypes.c_int,     # m
            ctypes.c_int,     # k
            ctypes.c_int      # n
        ]
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return

    print(f"\n📋 Creating GPU test data...")
    inputs = get_inputs()
    A, B = inputs
    output_torch = torch_kernel(A, B)
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    m, k = A.shape
    k2, n = B.shape
    
    # Create output tensor (float32 for GEMM)
    output_cuda = torch.empty(m, n, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(A_ptr, B_ptr, output_ptr, m, k, n)
    compare_results(output_torch, output_cuda, atol=1e-2, rtol=1e-2)  # Relaxed tolerance for mixed precision
    run_performance_test(A, B, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
