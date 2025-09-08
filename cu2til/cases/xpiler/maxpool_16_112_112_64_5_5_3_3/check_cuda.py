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
from cu2til.tools.layout import convert_nhwc_to_nchw

def get_inputs():
    """Create test data"""
    torch.manual_seed(SEED)
    # Generate data in NHWC format directly for CUDA kernel: (batch_size, height, width, channels)
    shape_nhwc = (16, 112, 112, 64)  # NHWC format
    
    # Create data directly on specified device in NHWC format
    x_nhwc = torch.randn(shape_nhwc, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    return x_nhwc

def run_performance_test(x_nhwc, cuda_kernel, kernel_size, stride):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, (x_nhwc, kernel_size, stride))
    
    """Call CUDA avgpool kernel - directly use GPU tensor"""
    batch_size, input_h, input_w, channels = x_nhwc.shape
    
    # Convert input from NCHW to NHWC for CUDA kernel (for performance test we need to convert)
    x_nhwc = convert_nhwc_to_nchw(x_nhwc)
    
    # Calculate output dimensions
    output_h = (input_h - kernel_size) // stride + 1
    output_w = (input_w - kernel_size) // stride + 1
    output_shape_nhwc = (batch_size, output_h, output_w, channels)  # NHWC format
    
    # Create output tensor in NHWC format
    output_gpu_nhwc = torch.empty(output_shape_nhwc, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    x_ptr = x_nhwc.data_ptr()
    output_ptr = output_gpu_nhwc.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (x_ptr, output_ptr, batch_size, channels, input_h, kernel_size, stride))
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 MAXPOOL CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename)
    shape = (16, 112, 112, 64)  # (batch_size, height, width, channels)
    batch_size, input_h, input_w, channels = shape
    kernel_size = 5
    stride = 3
    print(f"📊 Test parameters: shape={shape}, kernel_size={kernel_size}, stride={stride}")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # input (GPU pointer)
            ctypes.c_void_p,  # output (GPU pointer)
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # channels
            ctypes.c_int,     # input_H
            ctypes.c_int,     # kernel_size
            ctypes.c_int      # stride
        ]
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return

    print(f"\n📋 Creating GPU test data...")
    x_nhwc = get_inputs()
    x_nchw = convert_nhwc_to_nchw(x_nhwc)
    output_torch = torch_kernel(x_nchw, kernel_size, stride)
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    batch_size, input_h, input_w, channels = x_nhwc.shape
    
    # Calculate output dimensions
    output_h = (input_h - kernel_size) // stride + 1
    output_w = (input_w - kernel_size) // stride + 1
    output_shape_nhwc = (batch_size, output_h, output_w, channels)  # NHWC format
    
    # Create output tensor in NHWC format
    output_cuda_nhwc = torch.empty(output_shape_nhwc, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers (use pre-generated NHWC data)
    x_ptr = x_nhwc.data_ptr()
    output_ptr = output_cuda_nhwc.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(x_ptr, output_ptr, batch_size, channels, input_h, kernel_size, stride)
    
    # Convert output back from NHWC to NCHW for comparison
    output_cuda = convert_nhwc_to_nchw(output_cuda_nhwc)
    compare_results(output_torch, output_cuda, atol=1e-4)
    run_performance_test(x_nhwc, cuda_kernel, kernel_size, stride)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
