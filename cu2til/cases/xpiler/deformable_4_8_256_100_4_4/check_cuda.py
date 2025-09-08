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
    # Deformable Attention: deformable_4_8_256_100_4_4
    # This is a simplified implementation for the complex deformable attention operation
    batch_size = 4
    num_heads = 8
    embed_dim = 256
    num_queries = 100
    num_levels = 4
    num_points = 4
    
    # Create simplified input data (GPU tensors)
    value = torch.randn(batch_size, num_queries, num_heads, embed_dim // num_heads, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    value_spatial_shapes = torch.tensor([[20, 20], [10, 10]], dtype=torch.int32, device="cuda")  # Example spatial shapes
    level_start_index = torch.tensor([0, 400], dtype=torch.int32, device="cuda")  # Example level indices
    sampling_locations = torch.randn(batch_size, num_queries, num_heads, num_levels, num_points, 2, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    attention_weights = torch.randn(batch_size, num_queries, num_heads, num_levels, num_points, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    return (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)

def run_performance_test(inputs, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, inputs)
    
    """Call CUDA Deformable kernel - simplified version"""
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
    
    # Create output tensor
    output_gpu = torch.empty_like(value, device="cuda")
    
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
    
    # Parameter settings (inferred from filename)
    batch_size = 4
    num_heads = 8
    embed_dim = 256
    num_queries = 100
    num_levels = 4
    num_points = 4
    print(f"📊 Test parameters: Deformable Attention - batch={batch_size}, heads={num_heads}, embed_dim={embed_dim}, queries={num_queries}")
    
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
    
    # Create output tensor
    output_cuda = torch.empty_like(value, device="cuda")
    
    # Get GPU pointers
    value_ptr = value.cuda().contiguous().data_ptr()
    value_spatial_shapes_ptr = value_spatial_shapes.cuda().contiguous().data_ptr()
    level_start_index_ptr = level_start_index.cuda().contiguous().data_ptr()
    sampling_locations_ptr = sampling_locations.cuda().contiguous().data_ptr()
    attention_weights_ptr = attention_weights.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(value_ptr, value_spatial_shapes_ptr, level_start_index_ptr, sampling_locations_ptr, attention_weights_ptr, output_ptr)
    compare_results(output_torch, output_cuda, atol=1e-2)  # 大容差用于复杂操作
    run_performance_test(inputs, cuda_kernel, num_runs=3)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch.flatten()[:5]}")
    print(f"   CUDA:    {output_cuda.flatten()[:5]}")
    
if __name__ == "__main__":
    main()
