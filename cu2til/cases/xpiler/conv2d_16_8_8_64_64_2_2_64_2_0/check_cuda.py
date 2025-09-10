import torch
import numpy as np
import ctypes
import os
import sys
import argparse
from dataclasses import dataclass
from cu2til.tools.builder import SEED, load_cuda_kernel
from torch_.ref import torch_kernel
from cu2til.tools.checker import compare_results, run_performance_test

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, TESTCASE_ROOT_DIR)

@dataclass
class Conv2DParams:
    """Conv2D 参数配置"""
    batch_size: int = 16
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
    """生成输入数据（与新版本保持一致的NCHW格式）"""
    torch.manual_seed(SEED)
    input_tensor = torch.randn(params.batch_size, params.input_channels, params.input_height, params.input_width, 
                              dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    kernel_tensor = torch.randn(params.output_channels, params.input_channels, params.kernel_height, params.kernel_width, 
                               dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).to(memory_format=torch.channels_last)
    return input_tensor, kernel_tensor

def get_cuda_argtypes():
    """获取CUDA kernel的参数类型"""
    return [
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

def check_cuda_vs_torch(testname="Conv2D", enable_perf=False, compile_only=False):
    print(f"🚀 {testname} CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return False
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename) 
    params = Conv2DParams()
    print(f"📊 Test parameters: {testname} N={params.batch_size}, C={params.input_channels}, H={params.input_height}, W={params.input_width} -> N={params.batch_size}, O={params.output_channels}, oH={params.output_height}, oW={params.output_width}")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = get_cuda_argtypes()
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return False

    if compile_only:
        print(f"🔧 Compile-only mode: CUDA kernel compilation completed successfully")
        return True

    print(f"\n📋 Creating GPU test data...")
    input_tensor, kernel_tensor = get_inputs(params)  # NCHW format
    
    # Get PyTorch reference result
    output_torch = torch_kernel(input_tensor, kernel_tensor)
    
    # Run CUDA implementation
    print(f"⚡ Running CUDA kernel...")
    
    # Create output tensor (NCHW format for CUDA)
    output_cuda = torch.empty(params.batch_size, params.output_channels, params.output_height, params.output_width, 
                              dtype=torch.float32, device="cuda").to(memory_format=torch.channels_last)
    
    # Get GPU pointers
    input_ptr = input_tensor.data_ptr()
    kernel_ptr = kernel_tensor.data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(input_ptr, kernel_ptr, output_ptr, params.batch_size, params.input_height, 
               params.input_channels, params.output_channels, params.kernel_height, params.stride)
    
    # Compare results
    checkok = compare_results(output_torch, output_cuda, atol=1e-2, rtol=1e-2)  # 进一步放宽容差用于conv2d浮点精度
    
    # Performance testing if enabled
    if enable_perf:
        cuda_inputs_for_perf = [input_ptr, kernel_ptr, output_ptr, params.batch_size, params.input_height, 
                               params.input_channels, params.output_channels, params.kernel_height, params.stride]
        torch_inputs_for_perf = [input_tensor, kernel_tensor]
        run_performance_test(cuda_inputs_for_perf, torch_inputs_for_perf, cuda_kernel, torch_kernel, test_type=["CUDA", "PyTorch"])
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
    return checkok

def parse_args():
    parser = argparse.ArgumentParser(description='CUDA Conv2D kernel test')
    parser.add_argument('--compile-only', action='store_true', 
                       help='Only compile the CUDA kernel without running tests (default: False)')
    parser.add_argument('--no-perf', action='store_true',
                       help='Disable performance testing (default: False, performance testing enabled)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf  # Default is True, disable with --no-perf
    compile_only = args.compile_only  # Default is False
    
    # print(f"📋 Options: compile_only={compile_only}, enable_perf={enable_perf}")
    check_cuda_vs_torch(testname="Conv2D", enable_perf=enable_perf, compile_only=compile_only)