import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import time
import ctypes
import subprocess

# 导入我们创建的MSDeformAttnFunction版本
from msdeform_version import msdeform_kernel
# 导入原始的PyTorch参考实现  
from ref_naive import torch_kernel

# 导入CUDA编译工具
import sys
sys.path.insert(0, '/workspace/cu2til')
from cu2til.tools.builder import compile_cuda_kernel, CUDA_FOLDER_NAME, SEED, load_cuda_kernel

TESTCASE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # 上一级目录

def get_test_inputs():
    """创建与实际测试案例匹配的测试数据"""
    torch.manual_seed(SEED)  # 使用相同的随机种子确保一致性
    device = "cuda"
    
    # 实际测试案例参数
    lq = 100  # num_queries  
    m = 8     # num_heads
    d = 256   # embed_dim
    l = 4     # num_levels
    k = 4     # num_points per level
    
    # 定义4个level的空间形状 (height, width)
    spatial_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int64, device=device)
    
    # 计算每个level的起始索引
    level_starts = []
    total_spatial = 0
    for h, w in spatial_shapes:
        level_starts.append(total_spatial)
        total_spatial += h * w
    level_start_index = torch.tensor(level_starts, dtype=torch.int64, device=device)
    
    # 创建value tensor: [total_spatial_size, m, d]
    # total_spatial = 32*32 + 16*16 + 8*8 + 4*4 = 1024 + 256 + 64 + 16 = 1360
    value = torch.randn(total_spatial, m, d, dtype=torch.float32, device=device).normal_(mean=0.0, std=0.5)
    
    # 创建采样位置: [lq, m, l*k, 2] (归一化坐标 0-1)
    sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device=device)
    # 将坐标值限制在合理范围内
    sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
    
    # 创建注意力权重: [lq, m, l*k]  
    attention_weights = torch.randn(lq, m, l*k, dtype=torch.float32, device=device).normal_(mean=0.0, std=0.5)
    # 确保权重为正值并归一化
    attention_weights = torch.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    return (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)

def run_cuda_kernel(inputs):
    """运行原始的CUDA kernel"""
    try:
        # 编译CUDA kernel
        print("🔨 Compiling CUDA kernel...")
        compile_cuda_kernel(TESTCASE_ROOT_DIR)
        
        # 加载CUDA kernel
        argtypes = [
            ctypes.c_void_p,  # value
            ctypes.c_void_p,  # value_spatial_shapes
            ctypes.c_void_p,  # level_start_index
            ctypes.c_void_p,  # sampling_locations
            ctypes.c_void_p,  # attention_weights
            ctypes.c_void_p   # output
        ]
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes)
        
        value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
        
        # 创建输出张量: [lq, m, d] = [100, 8, 256]
        lq, m, d = 100, 8, 256
        output_gpu = torch.empty(lq, m, d, dtype=torch.float32, device="cuda")
        
        # 获取GPU指针
        value_ptr = value.cuda().contiguous().data_ptr()
        value_spatial_shapes_ptr = value_spatial_shapes.int().cuda().contiguous().data_ptr()
        level_start_index_ptr = level_start_index.int().cuda().contiguous().data_ptr()
        sampling_locations_ptr = sampling_locations.cuda().contiguous().data_ptr()
        attention_weights_ptr = attention_weights.cuda().contiguous().data_ptr()
        output_ptr = output_gpu.cuda().contiguous().data_ptr()
        
        # 调用CUDA kernel
        cuda_kernel(
            ctypes.c_void_p(value_ptr),
            ctypes.c_void_p(value_spatial_shapes_ptr),
            ctypes.c_void_p(level_start_index_ptr),
            ctypes.c_void_p(sampling_locations_ptr),
            ctypes.c_void_p(attention_weights_ptr),
            ctypes.c_void_p(output_ptr)
        )
        
        torch.cuda.synchronize()
        return output_gpu, True
        
    except Exception as e:
        print(f"❌ CUDA kernel failed: {e}")
        return None, False

def compare_outputs(output1, output2, name1="Output1", name2="Output2"):
    """比较两个输出的差异"""
    if output1.shape != output2.shape:
        print(f"❌ Shape mismatch: {name1} {output1.shape} vs {name2} {output2.shape}")
        return False
    
    abs_diff = torch.abs(output1 - output2)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    
    rel_diff = abs_diff / (torch.abs(output2) + 1e-8)
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()
    
    print(f"📊 {name1} vs {name2}:")
    print(f"  - Max absolute difference: {max_abs_diff:.2e}")
    print(f"  - Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"  - Max relative difference: {max_rel_diff:.2e}")
    print(f"  - Mean relative difference: {mean_rel_diff:.2e}")
    
    # 判断是否匹配 (允许一定的数值误差)
    is_close = max_abs_diff < 1e-3 and max_rel_diff < 1e-2
    print(f"  - {'✅ MATCH' if is_close else '❌ MISMATCH'}")
    
    return is_close

def benchmark_function(func, inputs, name, iterations=10):
    """对函数进行性能测试"""
    print(f"\n🚀 Benchmarking {name}...")
    
    # 预热
    for _ in range(3):
        if name == "CUDA Kernel":
            _ = func(inputs)
        else:
            _ = func(*inputs)
    
    torch.cuda.synchronize()
    
    # 计时
    times = []
    for i in range(iterations):
        start_time = time.time()
        if name == "CUDA Kernel":
            result = func(inputs)
        else:
            result = func(*inputs)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    print(f"  - Average time: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"  - Min time: {min_time:.3f} ms")
    
    return avg_time, min_time

def main():
    """主测试函数"""
    print("🧪 Deformable Attention: MSDeformAttnFunction vs CUDA Kernel")
    print("=" * 70)
    
    # 获取测试输入
    inputs = get_test_inputs()
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
    
    print(f"✅ Test inputs created:")
    print(f"  - value: {value.shape}")
    print(f"  - spatial_shapes: {value_spatial_shapes.shape}")
    print(f"  - level_start_index: {level_start_index.shape}")
    print(f"  - sampling_locations: {sampling_locations.shape}")
    print(f"  - attention_weights: {attention_weights.shape}")
    
    print("\n" + "=" * 70)
    print("📈 Running implementations...")
    
    # 运行MSDeformAttnFunction版本
    try:
        print("\n1️⃣ MSDeformAttnFunction Implementation:")
        msdeform_output = msdeform_kernel(*inputs)
        print(f"   Output shape: {msdeform_output.shape}")
        msdeform_success = True
    except Exception as e:
        print(f"❌ MSDeformAttnFunction failed: {e}")
        msdeform_output = None
        msdeform_success = False
    
    # 运行CUDA kernel
    print("\n2️⃣ CUDA Kernel Implementation:")
    cuda_output, cuda_success = run_cuda_kernel(inputs)
    if cuda_success:
        print(f"   Output shape: {cuda_output.shape}")
    
    print("\n" + "=" * 70)
    print("🔍 Accuracy comparison:")
    
    # 比较结果
    if msdeform_success and cuda_success:
        is_match = compare_outputs(msdeform_output, cuda_output, 
                                 "MSDeformAttnFunction", "CUDA Kernel")
        
        if is_match:
            print("\n🎉 MSDeformAttnFunction matches CUDA kernel results!")
        else:
            print("\n⚠️  MSDeformAttnFunction and CUDA kernel have differences!")
            
            # 显示一些样本值进行调试
            print(f"\nSample values (first 5 elements):")
            print(f"MSDeformAttnFunction: {msdeform_output.flatten()[:5]}")
            print(f"CUDA Kernel:          {cuda_output.flatten()[:5]}")
    else:
        print("❌ Cannot compare outputs due to implementation failures")
    
    print("\n" + "=" * 70)
    print("⚡ Performance comparison:")
    
    # 性能测试
    if msdeform_success:
        msdeform_avg, msdeform_min = benchmark_function(msdeform_kernel, inputs, "MSDeformAttnFunction", 20)
    
    if cuda_success:
        cuda_avg, cuda_min = benchmark_function(run_cuda_kernel, inputs, "CUDA Kernel", 20)
    
    if msdeform_success and cuda_success:
        speedup = cuda_avg / msdeform_avg
        print(f"\n🏆 MSDeformAttnFunction is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than CUDA kernel")
    
    print("\n" + "=" * 70)
    print("✨ Comparison completed!")
    
    # 总结
    print(f"\n📋 Summary:")
    print(f"  - MSDeformAttnFunction: {'✅ Working' if msdeform_success else '❌ Failed'}")
    print(f"  - CUDA Kernel: {'✅ Working' if cuda_success else '❌ Failed'}")
    if msdeform_success and cuda_success:
        print(f"  - Results match: {'✅ Yes' if is_match else '❌ No'}")
        print(f"  - Performance: MSDeformAttnFunction is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

if __name__ == "__main__":
    main()
