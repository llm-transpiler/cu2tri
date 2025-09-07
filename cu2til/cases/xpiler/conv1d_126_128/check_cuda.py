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
    # Conv1D: conv1d_126_128 -> input_size=128, output_size=126, kernel_size=3
    input_size = 128
    kernel_size = 3
    
    # Create data directly on specified device
    input_tensor = torch.randn(input_size, dtype=torch.float32, device="cuda")
    kernel_tensor = torch.randn(kernel_size, dtype=torch.float32, device="cuda")
    return input_tensor, kernel_tensor

def run_performance_test(input_tensor, kernel_tensor, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, (input_tensor, kernel_tensor))
    
    """Call CUDA Conv1D kernel - directly use GPU tensor"""
    input_size = input_tensor.size(0)
    output_size = 126
    
    # Ensure input tensors are on GPU and contiguous
    input_gpu = input_tensor.cuda().contiguous()
    kernel_gpu = kernel_tensor.cuda().contiguous()
    
    # Create output tensor
    output_gpu = torch.empty(output_size, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    input_ptr = input_gpu.data_ptr()
    kernel_ptr = kernel_gpu.data_ptr()
    output_ptr = output_gpu.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (input_ptr, kernel_ptr, output_ptr, input_size, output_size))
    
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 CONV1D CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename)
    input_size = 128
    output_size = 126
    kernel_size = 3
    print(f"📊 Test parameters: Conv1D input_size={input_size}, output_size={output_size}, kernel_size={kernel_size}")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # input (GPU pointer)
            ctypes.c_void_p,  # kernel (GPU pointer)
            ctypes.c_void_p,  # output (GPU pointer)
            ctypes.c_int,     # input_size
            ctypes.c_int      # output_size
        ]
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return

    print(f"\n📋 Creating GPU test data...")
    inputs = get_inputs()
    input_tensor, kernel_tensor = inputs
    output_torch = torch_kernel(input_tensor, kernel_tensor)
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    input_size = input_tensor.size(0)
    output_size = 126
    
    # Create output tensor
    output_cuda = torch.empty(output_size, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    input_ptr = input_tensor.cuda().contiguous().data_ptr()
    kernel_ptr = kernel_tensor.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(input_ptr, kernel_ptr, output_ptr, input_size, output_size)
    compare_results(output_torch, output_cuda, atol=1e-4)
    run_performance_test(input_tensor, kernel_tensor, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
