import torch
import numpy as np
import ctypes
import os
import time
import subprocess
import sys
from dataclasses import dataclass
from torch.nn import functional as F
from cu2til.tools.builder import compile_cuda_kernel, CUDA_FOLDER_NAME, SEED, load_cuda_kernel

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

import sys
sys.path.insert(0, TESTCASE_ROOT_DIR)

from torch_.ref import torch_kernel
from cu2til.tools.checker import compare_results

@dataclass
class DepthwiseConvParams:
    """DepthwiseConv 参数配置"""
    input_size: int = 192
    output_size: int = 190
    kernel_size: int = 3
    channels: int = 128

def get_inputs(params: DepthwiseConvParams):
    """Create test data"""
    torch.manual_seed(SEED)
    # DepthwiseConv: depthwiseconv_192_3_128 -> input({params.input_size}x{params.input_size}x{params.channels}), kernel({params.kernel_size}x{params.kernel_size}), output({params.output_size}x{params.output_size}x{params.channels})
    
    # Create data directly on specified device (H, W, C format)
    input_tensor = torch.randn(params.input_size, params.input_size, params.channels, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    # Depthwise kernel: (kernel_size, kernel_size, channels) - one filter per channel
    kernel_tensor = torch.randn(params.kernel_size, params.kernel_size, params.channels, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    return input_tensor, kernel_tensor

def run_performance_test(input_tensor, kernel_tensor, cuda_kernel, params: DepthwiseConvParams):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
        # 格式转换用于性能测试
    input_nchw = input_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    kernel_nchw = kernel_tensor.permute(2, 0, 1).unsqueeze(1).contiguous()
    torch_gpu_avg = benchmark_kernel(torch_kernel, (input_nchw, kernel_nchw))
    
    """Call CUDA DepthwiseConv kernel - directly use GPU tensor"""
    
    # Ensure input tensors are on GPU and contiguous
    input_gpu = input_tensor.cuda().contiguous()
    kernel_gpu = kernel_tensor.cuda().contiguous()
    
    # Create output tensor
    output_gpu = torch.empty(params.output_size, params.output_size, params.channels, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    input_ptr = input_gpu.data_ptr()
    kernel_ptr = kernel_gpu.data_ptr()
    output_ptr = output_gpu.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (input_ptr, kernel_ptr, output_ptr, params.input_size, params.kernel_size, params.channels))
    
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
    params = DepthwiseConvParams()
    print(f"📊 Test parameters: DepthwiseConv input({params.input_size}x{params.input_size}x{params.channels}), kernel({params.kernel_size}x{params.kernel_size}), output({params.output_size}x{params.output_size}x{params.channels})")
    
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
    input_tensor, kernel_tensor = get_inputs(params)
        # 格式转换：HWC -> NCHW for PyTorch
    # Input: (H, W, C) -> (1, C, H, W)
    input_nchw = input_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    # Kernel: (kH, kW, C) -> (C, 1, kH, kW) for depthwise conv  
    kernel_nchw = kernel_tensor.permute(2, 0, 1).unsqueeze(1).contiguous()
    
    # 调用torch_kernel进行纯计算
    output_nchw = torch_kernel(input_nchw, kernel_nchw)
    
    # 转换回原格式：(1, C, H, W) -> (H, W, C)
    output_torch = output_nchw.squeeze(0).permute(1, 2, 0).contiguous()
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    
    # Create output tensor
    output_cuda = torch.empty(params.output_size, params.output_size, params.channels, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    input_ptr = input_tensor.cuda().contiguous().data_ptr()
    kernel_ptr = kernel_tensor.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(input_ptr, kernel_ptr, output_ptr, params.input_size, params.kernel_size, params.channels)
    compare_results(output_torch, output_cuda, atol=1e-4)
    run_performance_test(input_tensor, kernel_tensor, cuda_kernel, params)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
