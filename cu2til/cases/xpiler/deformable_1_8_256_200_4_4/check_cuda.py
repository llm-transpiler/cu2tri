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
    """Create test data - 匹配CUDA kernel的数据格式"""
    torch.manual_seed(SEED)
    
    # 从CUDA kernel分析得到的参数
    lq = 200  # num_queries  
    m = 8     # num_heads
    d = 256   # embed_dim
    l = 4     # num_levels
    k = 4     # num_points per level
    
    # 定义4个level的空间形状 (height, width)
    spatial_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int32, device="cuda")
    
    # 计算每个level的起始索引
    level_starts = []
    total_spatial = 0
    for h, w in spatial_shapes:
        level_starts.append(total_spatial)
        total_spatial += h * w
    level_start_index = torch.tensor(level_starts, dtype=torch.int32, device="cuda")
    
    # 创建value tensor: [total_spatial_size, m, d]
    # total_spatial = 32*32 + 16*16 + 8*8 + 4*4 = 1024 + 256 + 64 + 16 = 1360
    value = torch.randn(total_spatial, m, d, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # 创建采样位置: [lq, m, l*k, 2] (归一化坐标 0-1)
    sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device="cuda")
    # 将坐标值限制在合理范围内
    sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
    
    # 创建注意力权重: [lq, m, l*k]  
    attention_weights = torch.randn(lq, m, l*k, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    # 确保权重为正值并归一化
    attention_weights = torch.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    return (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)

def run_performance_test(inputs, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, inputs)
    
    """Call CUDA Deformable kernel - simplified version"""
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
    
    # Create output tensor: [lq, m, d] = [200, 8, 256]
    lq, m, d = 200, 8, 256
    output_gpu = torch.empty(lq, m, d, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers - simplified parameter passing
    value_ptr = value.cuda().contiguous().data_ptr()
    value_spatial_shapes_ptr = value_spatial_shapes.cuda().contiguous().data_ptr()
    level_start_index_ptr = level_start_index.cuda().contiguous().data_ptr()
    sampling_locations_ptr = sampling_locations.cuda().contiguous().data_ptr()
    attention_weights_ptr = attention_weights.cuda().contiguous().data_ptr()
    output_ptr = output_gpu.data_ptr()
    
    # Note: This is a simplified interface - actual deformable attention is very complex
    cuda_avg = benchmark_kernel(cuda_kernel, (value_ptr, value_spatial_shapes_ptr, level_start_index_ptr, sampling_locations_ptr, attention_weights_ptr, output_ptr))
    
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 DEFORMABLE CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (从CUDA kernel分析得到)
    lq = 200  # num_queries
    m = 8     # num_heads  
    d = 256   # embed_dim
    l = 4     # num_levels
    k = 4     # num_points per level
    print(f"📊 Test parameters: Deformable Attention - queries={lq}, heads={m}, embed_dim={d}, levels={l}, points={k}")
    print(f"   Data shapes: value=[total_spatial, {m}, {d}], sampling_locations=[{lq}, {m}, {l*k}, 2], attention_weights=[{lq}, {m}, {l*k}]")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # value
            ctypes.c_void_p,  # value_spatial_shapes
            ctypes.c_void_p,  # level_start_index
            ctypes.c_void_p,  # sampling_locations
            ctypes.c_void_p,  # attention_weights
            ctypes.c_void_p   # output
        ]
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return

    print(f"\n📋 Creating GPU test data...")
    inputs = get_inputs()
    output_torch = torch_kernel(*inputs)
    
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
    
    # Create output tensor: [lq, m, d] = [200, 8, 256]  
    lq, m, d = 200, 8, 256
    output_cuda = torch.empty(lq, m, d, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers
    value_ptr = value.cuda().contiguous().data_ptr()
    value_spatial_shapes_ptr = value_spatial_shapes.cuda().contiguous().data_ptr()
    level_start_index_ptr = level_start_index.cuda().contiguous().data_ptr()
    sampling_locations_ptr = sampling_locations.cuda().contiguous().data_ptr()
    attention_weights_ptr = attention_weights.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(value_ptr, value_spatial_shapes_ptr, level_start_index_ptr, sampling_locations_ptr, attention_weights_ptr, output_ptr)
    compare_results(output_torch, output_cuda, atol=1e-2, rtol=1e-2)  # 大容差用于复杂操作
    run_performance_test(inputs, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
