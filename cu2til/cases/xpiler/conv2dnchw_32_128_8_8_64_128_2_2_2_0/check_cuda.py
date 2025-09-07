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
    # Conv2DNCHW: conv2dnchw_32_128_8_8_64_128_2_2_2_0 (NCHW format)
    # Input: (N, C, H, W) = (32, 128, 8, 8)
    # Kernel: (O, C, kH, kW) = (64, 128, 2, 2)
    # Output: (N, O, oH, oW) = (32, 64, 4, 4)
    
    # Create data directly on specified device (NCHW format)
    input_tensor = torch.randn(32, 128, 8, 8, dtype=torch.float32, device="cuda")
    kernel_tensor = torch.randn(64, 128, 2, 2, dtype=torch.float32, device="cuda")
    return input_tensor, kernel_tensor

def run_performance_test(input_tensor, kernel_tensor, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, (input_tensor, kernel_tensor))
    
    """Call CUDA Conv2DNCHW kernel - directly use GPU tensor"""
    batch_size, input_channels, input_height, input_width = input_tensor.shape
    output_channels, _, kernel_height, kernel_width = kernel_tensor.shape
    stride = 2
    
    # Ensure input tensors are on GPU and contiguous
    input_gpu = input_tensor.cuda().contiguous()
    kernel_gpu = kernel_tensor.cuda().contiguous()
    
    # Create output tensor
    output_height = 4
    output_width = 4
    output_gpu = torch.empty(batch_size, output_channels, output_height, output_width, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    input_ptr = input_gpu.data_ptr()
    kernel_ptr = kernel_gpu.data_ptr()
    output_ptr = output_gpu.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (input_ptr, kernel_ptr, output_ptr, batch_size, input_height, input_channels, output_channels, kernel_height, stride))
    
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 CONV2DNCHW CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename)
    batch_size = 32
    input_channels = 128
    input_height, input_width = 8, 8
    output_channels = 64
    kernel_height, kernel_width = 2, 2
    stride, padding = 2, 0
    output_height, output_width = 4, 4
    print(f"📊 Test parameters: Conv2DNCHW N={batch_size}, C={input_channels}, H={input_height}, W={input_width} -> N={batch_size}, O={output_channels}, oH={output_height}, oW={output_width}")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # input (GPU pointer)
            ctypes.c_void_p,  # kernel (GPU pointer)
            ctypes.c_void_p,  # output (GPU pointer)
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # input_height
            ctypes.c_int,     # input_channels
            ctypes.c_int,     # output_channels
            ctypes.c_int,     # kernel_height
            ctypes.c_int      # stride
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
    batch_size, input_channels, input_height, input_width = input_tensor.shape
    output_channels, _, kernel_height, kernel_width = kernel_tensor.shape
    stride = 2
    
    # Create output tensor
    output_height = 4
    output_width = 4
    output_cuda = torch.empty(batch_size, output_channels, output_height, output_width, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    input_ptr = input_tensor.cuda().contiguous().data_ptr()
    kernel_ptr = kernel_tensor.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(input_ptr, kernel_ptr, output_ptr, batch_size, input_height, input_channels, output_channels, kernel_height, stride)
    compare_results(output_torch, output_cuda, atol=3e-2, rtol=1e-2)  # Relaxed tolerance for conv2dnchw
    run_performance_test(input_tensor, kernel_tensor, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
