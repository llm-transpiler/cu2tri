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
    # Shape inferred from filename: rmsnorm_4096_4096 -> [4096, 4096]
    shape = (4096, 4096)
    
    # Create data directly on specified device
    x = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    return x,

def run_performance_test(x, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, (x,))
    
    """Call CUDA rmsnorm kernel - directly use GPU tensor"""
    size_1, size_2 = x.shape
    
    # Ensure input tensor is on GPU and contiguous
    x_gpu = x.cuda().contiguous()
    
    # Create output tensor
    output_gpu = torch.empty_like(x_gpu)
    
    # Get GPU pointers
    x_ptr = x_gpu.data_ptr()
    output_ptr = output_gpu.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (x_ptr, output_ptr, size_1, size_2))
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 RMSNORM CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename)
    shape = (4096, 4096)
    size_1, size_2 = shape
    print(f"📊 Test parameters: shape={shape}, size_1={size_1}, size_2={size_2}")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # input (GPU pointer)
            ctypes.c_void_p,  # output (GPU pointer)
            ctypes.c_int,     # size_1
            ctypes.c_int      # size_2
        ]
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return

    print(f"\n📋 Creating GPU test data...")
    inputs = get_inputs()
    x = inputs[0]
    output_torch = torch_kernel(x)
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    size_1, size_2 = x.shape
    # Create output tensor
    output_cuda = torch.empty_like(x)
    
    # Get GPU pointers
    x_ptr = x.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(x_ptr, output_ptr, size_1, size_2)
    compare_results(output_torch, output_cuda, atol=1e-4)
    run_performance_test(x, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
