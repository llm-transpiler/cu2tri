import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    """Create test data - 兼容CUDA kernel和新ref.py的数据格式"""
    torch.manual_seed(SEED)
    
    # 从CUDA kernel分析得到的参数
    lq = 100  # num_queries  
    m = 8     # num_heads
    d = 256   # embed_dim
    l = 4     # num_levels
    k = 4     # num_points per level
    
    # 定义4个level的空间形状 (height, width)
    spatial_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    # 使用int64以兼容新的ref.py（官方实现）
    value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int64, device="cuda")
    
    # 计算每个level的起始索引
    level_starts = []
    total_spatial = 0
    for h, w in spatial_shapes:
        level_starts.append(total_spatial)
        total_spatial += h * w
    # 使用int64以兼容新的ref.py
    level_start_index = torch.tensor(level_starts, dtype=torch.int64, device="cuda")
    
    # 创建value tensor: [total_spatial_size, m, d]
    # total_spatial = 32*32 + 16*16 + 8*8 + 4*4 = 1024 + 256 + 64 + 16 = 1360
    # 恢复与CUDA kernel对齐的初始化方式
    value = torch.randn(total_spatial, m, d, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # 创建采样位置: [lq, m, l*k, 2] (归一化坐标 0-1)
    sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device="cuda")
    # 将坐标值限制在合理范围内
    sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
    
    # 创建注意力权重: [lq, m, l*k] - 恢复与CUDA kernel对齐的初始化方式
    attention_weights = torch.randn(lq, m, l*k, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    # 确保权重为正值并归一化（与ref_naive.py和CUDA kernel对齐）
    attention_weights = torch.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    print(f"📊 Generated inputs:")
    print(f"   - value: {value.shape} ({value.dtype})")
    print(f"   - value_spatial_shapes: {value_spatial_shapes.shape} ({value_spatial_shapes.dtype})")
    print(f"   - level_start_index: {level_start_index.shape} ({level_start_index.dtype})")
    print(f"   - sampling_locations: {sampling_locations.shape} ({sampling_locations.dtype})")
    print(f"   - attention_weights: {attention_weights.shape} ({attention_weights.dtype})")
    
    return (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)

def run_performance_test(inputs, cuda_kernel):
    """Run GPU performance test with updated PyTorch reference implementation"""
    print(f"\n🚀 GPU performance test:")
    
    try:
        from eval_.common.benchmark import _simple_perf
        
        # Test PyTorch implementation (新的官方参考实现)
        print("🧪 Benchmarking 官方PyTorch参考实现...")
        torch_gpu_avg = _simple_perf(torch_kernel, inputs, warmup=2, iterations=5)
        torch_gpu_avg = torch_gpu_avg["median"]
        
        """Call CUDA Deformable kernel with correct data type conversion"""
        value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
        
        # Create output tensor: [lq, m, d] = [100, 8, 256]
        lq, m, d = 100, 8, 256
        output_gpu = torch.empty(lq, m, d, dtype=torch.float32, device="cuda")
        
        # Get GPU pointers - convert int64 to int32 for CUDA kernel compatibility
        value_ptr = value.cuda().contiguous().data_ptr()
        value_spatial_shapes_ptr = value_spatial_shapes.int().cuda().contiguous().data_ptr()  # int64 -> int32
        level_start_index_ptr = level_start_index.int().cuda().contiguous().data_ptr()        # int64 -> int32
        sampling_locations_ptr = sampling_locations.cuda().contiguous().data_ptr()
        attention_weights_ptr = attention_weights.cuda().contiguous().data_ptr()
        output_ptr = output_gpu.data_ptr()
        
        # Test CUDA kernel implementation  
        print("🧪 Benchmarking CUDA kernel...")
        try:
            cuda_avg = _simple_perf(cuda_kernel, (value_ptr, value_spatial_shapes_ptr, level_start_index_ptr, sampling_locations_ptr, attention_weights_ptr, output_ptr), warmup=2, iterations=5)
            cuda_avg = cuda_avg["median"]
            
            print(f"\n📊 GPU performance comparison:")
            print(f"  官方PyTorch参考实现: {torch_gpu_avg:7.3f} ms")
            print(f"  CUDA kernel:         {cuda_avg:7.3f} ms")
            
            if cuda_avg > 0:
                speedup = torch_gpu_avg / cuda_avg
                print(f"  CUDA speedup:          {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
            
            return torch_gpu_avg, cuda_avg
            
        except Exception as e:
            print(f"❌ CUDA kernel benchmark failed: {e}")
            print(f"📊 只有PyTorch参考实现的性能:")
            print(f"  官方PyTorch参考实现: {torch_gpu_avg:7.3f} ms")
            return torch_gpu_avg, None
            
    except ImportError:
        print("⚠️  Benchmark module not available, skipping performance test")
        return None, None
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return None, None

def main():
    print("🚀 DEFORMABLE CUDA automatic test (使用官方PyTorch参考实现)")
    print("=" * 65)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (从CUDA kernel分析得到)
    lq = 100  # num_queries
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

    print(f"\n📋 Creating GPU test data (与CUDA kernel对齐的格式)...")
    inputs = get_inputs()
    print(f"\n🧪 Running 官方PyTorch参考实现 (ms_deform_attn_core_pytorch)...")
    output_torch = torch_kernel(*inputs)
    print(f"✅ 官方PyTorch参考实现成功，输出形状: {output_torch.shape}")
    print(f"   输出统计: mean={output_torch.mean().item():.6f}, std={output_torch.std().item():.6f}")
    
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
    
    # Create output tensor: [lq, m, d] = [100, 8, 256]  
    lq, m, d = 100, 8, 256
    output_cuda = torch.empty(lq, m, d, dtype=torch.float32, device="cuda")
    
    # Get GPU pointers - CUDA kernel expects int32, so convert from int64
    value_ptr = value.cuda().contiguous().data_ptr()
    value_spatial_shapes_ptr = value_spatial_shapes.int().cuda().contiguous().data_ptr()  # int64 -> int32
    level_start_index_ptr = level_start_index.int().cuda().contiguous().data_ptr()        # int64 -> int32  
    sampling_locations_ptr = sampling_locations.cuda().contiguous().data_ptr()
    attention_weights_ptr = attention_weights.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    try:
        # Call CUDA kernel
        cuda_kernel(value_ptr, value_spatial_shapes_ptr, level_start_index_ptr, sampling_locations_ptr, attention_weights_ptr, output_ptr)
        print(f"✅ CUDA kernel executed successfully")
        torch.cuda.synchronize()  # 确保CUDA操作完成
    except Exception as e:
        print(f"❌ CUDA kernel failed: {e}")
        print("🔄 This may be due to the hardcoded parameters issue in the CUDA kernel")
        return
    compare_results(output_torch, output_cuda, atol=1e-2, rtol=1e-2)  # 大容差用于复杂操作
    run_performance_test(inputs, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   官方PyTorch参考实现: {output_torch.flatten()[:5]}")
    print(f"   CUDA kernel:        {output_cuda.flatten()[:5]}")
    
    print(f"\n🎉 测试完成!")
    print(f"   - 使用了官方的 ms_deform_attn_core_pytorch 作为参考实现")
    print(f"   - 对比了官方实现与自定义CUDA kernel的结果")
    print(f"   - 如果CUDA kernel失败，推荐直接使用MSDeformAttnFunction")
    
if __name__ == "__main__":
    main()
