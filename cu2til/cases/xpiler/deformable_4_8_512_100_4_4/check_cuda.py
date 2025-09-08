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
    """Create correct batched test data - 匹配CUDA kernel的真实内存布局"""
    torch.manual_seed(SEED)
    
    # 从CUDA kernel分析得到的参数 (deformable_4_8_512_100_4_4)
    n = 4     # batch_size (kernel硬编码的batch数量)
    lq = 100  # num_queries  
    m = 8     # num_heads
    d = 512   # embed_dim
    l = 4     # num_levels
    k = 4     # num_points per level
    
    # 从kernel内存访问分析得到的正确spatial size
    kernel_value_stride = 53661696
    correct_spatial_size = kernel_value_stride // (m * d)  # 13101
    
    # 定义精确匹配13101的空间形状配置
    base_size = 114  # 114*114 = 12996
    remaining = correct_spatial_size - base_size * base_size  # 105
    spatial_shapes = [(base_size, base_size), (remaining-2, 1), (1, 1), (1, 1)]  # 12996 + 103 + 1 + 1 = 13101
    
    value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int32, device="cuda")
    
    # 计算每个level的起始索引
    level_starts = []
    total_spatial = 0
    for h, w in spatial_shapes:
        level_starts.append(total_spatial)
        total_spatial += h * w
    level_start_index = torch.tensor(level_starts, dtype=torch.int32, device="cuda")
    
    # 为了安全，使用kernel期望的确切大小
    actual_spatial_size = correct_spatial_size
    
    # 创建正确的batched数据
    value = torch.randn(n, actual_spatial_size, m, d, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    sampling_locations = torch.rand(n, lq, m, l*k, 2, dtype=torch.float32, device="cuda")
    sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
    
    attention_weights = torch.randn(n, lq, m, l*k, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    attention_weights = torch.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    # 为兼容性返回相同的数据两次
    return (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, 
            value, sampling_locations, attention_weights)

def run_performance_test(inputs_tuple, cuda_kernel):
    """Run GPU performance test with both batched and flattened data"""
    print(f"\n🚀 GPU performance test:")
    
    try:
        from eval_.common.benchmark import benchmark_kernel
        
        inputs_for_torch, inputs_for_cuda = inputs_tuple
        
        # Benchmark PyTorch with batched data
        torch_gpu_avg = benchmark_kernel(torch_kernel, inputs_for_torch)
        
        """Call CUDA Deformable kernel with correct batched input data"""
        value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs_for_cuda
        
        # Create batched output tensor: [n, lq, m, d] = [4, 100, 8, 512]
        n, lq, m, d = 4, 100, 8, 512
        output_gpu = torch.empty(n, lq, m, d, dtype=torch.float32, device="cuda")
        
        # Get GPU pointers - all inputs are properly batched
        value_ptr = value.cuda().contiguous().data_ptr()
        value_spatial_shapes_ptr = value_spatial_shapes.cuda().contiguous().data_ptr()
        level_start_index_ptr = level_start_index.cuda().contiguous().data_ptr()
        sampling_locations_ptr = sampling_locations.cuda().contiguous().data_ptr()
        attention_weights_ptr = attention_weights.cuda().contiguous().data_ptr()
        output_ptr = output_gpu.data_ptr()
        
        # Benchmark CUDA kernel with batched data
        cuda_avg = benchmark_kernel(cuda_kernel, (value_ptr, value_spatial_shapes_ptr, level_start_index_ptr, sampling_locations_ptr, attention_weights_ptr, output_ptr))
        
        print(f"\n📊 GPU performance comparison (n={n} batched):")
        print(f"  PyTorch (batched): {torch_gpu_avg:7.3f} ms")
        print(f"  CUDA kernel:       {cuda_avg:7.3f} ms")
        
        if cuda_avg > 0:
            speedup = torch_gpu_avg / cuda_avg
            print(f"  CUDA speedup:        {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
            
        return torch_gpu_avg, cuda_avg
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return None, None

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
    n = 4     # batch_size (kernel硬编码)
    lq = 100  # num_queries
    m = 8     # num_heads  
    d = 512   # embed_dim
    l = 4     # num_levels
    k = 4     # num_points per level
    spatial_size = 13101  # kernel期望的spatial size
    print(f"📊 Test parameters: Deformable Attention - batch={n}, queries={lq}, heads={m}, embed_dim={d}, levels={l}, points={k}")
    
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
    data = get_inputs()
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, value_batched, sampling_locations_batched, attention_weights_batched = data
    
    # 使用batched数据进行PyTorch参考实现
    inputs_for_torch = (value_batched, value_spatial_shapes, level_start_index, sampling_locations_batched, attention_weights_batched)
    output_torch = torch_kernel(*inputs_for_torch)
    
    # Run CUDA implementation with correct batched data
    print(f"\n⚡ Running CUDA kernel with correct batched data...")
    
    # Create BATCHED output tensor: [n, lq, m, d] = [4, 100, 8, 512]  
    n, lq, m, d = 4, 100, 8, 512
    output_cuda = torch.empty(n, lq, m, d, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers for batched data (所有tensor都有正确的batch维度)
    value_ptr = value.cuda().contiguous().data_ptr()
    value_spatial_shapes_ptr = value_spatial_shapes.cuda().contiguous().data_ptr()
    level_start_index_ptr = level_start_index.cuda().contiguous().data_ptr()
    sampling_locations_ptr = sampling_locations.cuda().contiguous().data_ptr()
    attention_weights_ptr = attention_weights.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(value_ptr, value_spatial_shapes_ptr, level_start_index_ptr, sampling_locations_ptr, attention_weights_ptr, output_ptr)
    
    print(f"   ✅ CUDA kernel执行完成！")
    compare_results(output_torch, output_cuda, atol=1e-2, rtol=1e-2)  # 大容差用于复杂操作
    # 为性能测试准备正确的输入格式
    inputs_for_perf = (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)
    run_performance_test((inputs_for_torch, inputs_for_perf), cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch (batched):  {output_torch.flatten()[:5]}")
    print(f"   CUDA (batched):     {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()