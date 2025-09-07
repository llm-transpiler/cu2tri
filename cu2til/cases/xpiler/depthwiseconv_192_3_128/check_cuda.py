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
    # DepthwiseConv: depthwiseconv_192_3_128 -> input(192x192x128), kernel(3x3), output(190x190x128)
    input_height, input_width = 192, 192
    channels = 128
    kernel_size = 3
    
    # Create data directly on specified device (H, W, C format)
    input_tensor = torch.randn(input_height, input_width, channels, dtype=torch.float32, device="cuda")
    # Depthwise kernel: (kernel_size, kernel_size, channels) - one filter per channel
    kernel_tensor = torch.randn(kernel_size, kernel_size, channels, dtype=torch.float32, device="cuda")
    return input_tensor, kernel_tensor

def run_performance_test(input_tensor, kernel_tensor, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, (input_tensor, kernel_tensor))
    
    """Call CUDA DepthwiseConv kernel - directly use GPU tensor"""
    input_height, input_width, input_channels = input_tensor.shape
    kernel_size = 3
    
    # Ensure input tensors are on GPU and contiguous
    input_gpu = input_tensor.cuda().contiguous()
    kernel_gpu = kernel_tensor.cuda().contiguous()
    
    # Create output tensor
    output_height = input_height - (kernel_size - 1)
    output_width = input_width - (kernel_size - 1)
    output_gpu = torch.empty(output_height, output_width, input_channels, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    input_ptr = input_gpu.data_ptr()
    kernel_ptr = kernel_gpu.data_ptr()
    output_ptr = output_gpu.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (input_ptr, kernel_ptr, output_ptr, input_height, kernel_size, input_channels))
    
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 DEPTHWISECONV CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename)
    input_size = 192
    output_size = 190
    kernel_size = 3
    channels = 128
    print(f"📊 Test parameters: DepthwiseConv input({input_size}x{input_size}x{channels}), kernel({kernel_size}x{kernel_size}), output({output_size}x{output_size}x{channels})")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # input (GPU pointer)
            ctypes.c_void_p,  # kernel (GPU pointer)
            ctypes.c_void_p,  # output (GPU pointer)
            ctypes.c_int,     # input_height
            ctypes.c_int,     # kernel_size
            ctypes.c_int      # input_channels
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
    input_height, input_width, input_channels = input_tensor.shape
    kernel_size = 3
    
    # Create output tensor
    output_height = input_height - (kernel_size - 1)
    output_width = input_width - (kernel_size - 1)
    output_cuda = torch.empty(output_height, output_width, input_channels, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    input_ptr = input_tensor.cuda().contiguous().data_ptr()
    kernel_ptr = kernel_tensor.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(input_ptr, kernel_ptr, output_ptr, input_height, kernel_size, input_channels)
    compare_results(output_torch, output_cuda, atol=1e-4)
    run_performance_test(input_tensor, kernel_tensor, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
