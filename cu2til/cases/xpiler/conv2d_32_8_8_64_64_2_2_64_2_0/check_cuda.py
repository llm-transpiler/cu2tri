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
class Conv2DParams:
    """Conv2D 参数配置"""
    batch_size: int = 32
    input_height: int = 8
    input_width: int = 8
    input_channels: int = 64
    output_channels: int = 64
    kernel_height: int = 2
    kernel_width: int = 2
    stride: int = 2
    padding: int = 0
    output_height: int = 4
    output_width: int = 4

def get_inputs(params: Conv2DParams):
    """Create test data"""
    torch.manual_seed(SEED)
    # Conv2D: conv2d_32_8_8_64_64_2_2_64_2_0
    # 以CUDA kernel的格式需求为准
    # Input: NHWC = (batch_size, input_height, input_width, input_channels) 
    # Kernel: OHWI = (output_channels, kernel_height, kernel_width, input_channels)
    
    # Create data directly on specified device (NHWC and OHWI format)
    input_tensor = torch.randn(params.batch_size, params.input_height, params.input_width, params.input_channels, 
                              dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    kernel_tensor = torch.randn(params.output_channels, params.kernel_height, params.kernel_width, params.input_channels, 
                               dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    return input_tensor, kernel_tensor

def run_performance_test(input_tensor, kernel_tensor, cuda_kernel, params: Conv2DParams):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    
    # 转换为PyTorch格式进行基准测试
    input_nchw = input_tensor.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    kernel_oihw = kernel_tensor.permute(0, 3, 1, 2).contiguous()  # OHWI -> OIHW
    torch_gpu_avg = benchmark_kernel(torch_kernel, (input_nchw, kernel_oihw))
    
    """Call CUDA Conv2D kernel - directly use GPU tensor"""
    
    # Create output tensor (NHWC format for CUDA)
    output_gpu_nhwc = torch.empty(params.batch_size, params.output_height, params.output_width, params.output_channels, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers (input已经是NHWC格式，无需转换)
    input_ptr = input_tensor.data_ptr()
    kernel_ptr = kernel_tensor.data_ptr()
    output_ptr = output_gpu_nhwc.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (input_ptr, kernel_ptr, output_ptr, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride))
    
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 CONV2D CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename) 
    params = Conv2DParams()
    print(f"📊 Test parameters: Conv2D N={params.batch_size}, C={params.input_channels}, H={params.input_height}, W={params.input_width} -> N={params.batch_size}, O={params.output_channels}, oH={params.output_height}, oW={params.output_width}")
    
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
    input_tensor, kernel_tensor = get_inputs(params)  # NHWC and OHWI format
    
    # 转换为PyTorch格式进行参考计算
    input_nchw = input_tensor.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    kernel_oihw = kernel_tensor.permute(0, 3, 1, 2).contiguous()  # OHWI -> OIHW
    output_torch = torch_kernel(input_nchw, kernel_oihw)
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    
    # Create output tensor (NHWC format for CUDA)
    output_cuda_nhwc = torch.empty(params.batch_size, params.output_height, params.output_width, params.output_channels, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers (input已经是NHWC格式，无需转换)
    input_ptr = input_tensor.data_ptr()
    kernel_ptr = kernel_tensor.data_ptr()
    output_ptr = output_cuda_nhwc.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(input_ptr, kernel_ptr, output_ptr, params.batch_size, params.input_height, params.input_channels, params.output_channels, params.kernel_height, params.stride)
    
    # 将CUDA输出从NHWC转换为NCHW格式与PyTorch结果比较
    output_cuda = output_cuda_nhwc.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    compare_results(output_torch, output_cuda, atol=1e-2)  # 放宽容差用于conv2d
    run_performance_test(input_tensor, kernel_tensor, cuda_kernel, params)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
