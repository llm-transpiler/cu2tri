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
    # GEMV operation: gemv_112_128
    # GEMV: A(112,128) @ x(128) = y(112)
    A = torch.randn(112, 128, dtype=torch.float32, device="cuda")
    x = torch.randn(128, dtype=torch.float32, device="cuda")
    return A, x

def run_performance_test(A, x, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, (A, x))
    
    """Call CUDA GEMV kernel - directly use GPU tensor"""
    m, n = A.shape
    
    # Create output vector
    y = torch.empty(m, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    A_ptr = A.data_ptr()
    x_ptr = x.data_ptr()
    y_ptr = y.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (A_ptr, x_ptr, y_ptr, m, n))
    
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 GEMV CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename)
    m, n = 112, 128
    print(f"📊 Test parameters: GEMV A({m},{n}) @ x({n}) = y({m})")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # A (float* GPU pointer)
            ctypes.c_void_p,  # x (float* GPU pointer)
            ctypes.c_void_p,  # y (float* GPU pointer)
            ctypes.c_int,     # m
            ctypes.c_int      # n
        ]
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return

    print(f"\n📋 Creating GPU test data...")
    inputs = get_inputs()
    A, x = inputs
    output_torch = torch_kernel(A, x)
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    m, n = A.shape
    
    # Create output vector
    output_cuda = torch.empty(m, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    A_ptr = A.data_ptr()
    x_ptr = x.data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(A_ptr, x_ptr, output_ptr, m, n)
    compare_results(output_torch, output_cuda, atol=1e-4)
    run_performance_test(A, x, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
