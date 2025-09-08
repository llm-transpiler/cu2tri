import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import time

# 导入我们创建的MSDeformAttnFunction版本
from msdeform_version import msdeform_kernel
# 导入官方的PyTorch参考实现
from ref import torch_kernel

def get_test_inputs():
    """创建与实际测试案例匹配的测试数据"""
    torch.manual_seed(42)
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
    
    inputs = (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)
    
    print(f"✅ Test inputs created:")
    print(f"  - value: {value.shape}")
    print(f"  - spatial_shapes: {value_spatial_shapes.shape} = {spatial_shapes}")
    print(f"  - level_start_index: {level_start_index.shape} = {level_starts}")
    print(f"  - sampling_locations: {sampling_locations.shape}")
    print(f"  - attention_weights: {attention_weights.shape}")
    print(f"  - total_spatial: {total_spatial}")
    
    return inputs

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
    is_close = torch.allclose(output1, output2, atol=1e-3, rtol=1e-2)
    print(f"  - {'✅ MATCH' if is_close else '❌ MISMATCH'}")
    
    return is_close

def benchmark_function(func, inputs, name, iterations=10):
    """对函数进行性能测试"""
    print(f"\n🚀 Benchmarking {name}...")
    
    # 预热
    for _ in range(3):
        _ = func(*inputs)
    
    torch.cuda.synchronize()
    
    # 计时
    times = []
    for i in range(iterations):
        start_time = time.time()
        _ = func(*inputs)
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
    print("🧪 Deformable Attention Implementation Comparison")
    print("=" * 60)
    
    # 获取测试输入
    inputs = get_test_inputs()
    
    print("\n" + "=" * 60)
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
    
    # 运行PyTorch参考实现
    try:
        print("\n2️⃣ PyTorch Reference Implementation:")
        torch_output = torch_kernel(*inputs)
        print(f"   Output shape: {torch_output.shape}")
        torch_success = True
    except Exception as e:
        print(f"❌ PyTorch reference failed: {e}")
        torch_output = None
        torch_success = False
    
    print("\n" + "=" * 60)
    print("🔍 Accuracy comparison:")
    
    # 比较结果
    if msdeform_success and torch_success:
        is_match = compare_outputs(msdeform_output, torch_output, 
                                 "MSDeformAttnFunction", "PyTorch Reference")
        
        if is_match:
            print("\n🎉 All implementations produce matching results!")
        else:
            print("\n⚠️  Implementations have significant differences!")
            
            # 显示一些样本值进行调试
            print(f"\nSample values (first 5 elements):")
            print(f"MSDeformAttnFunction: {msdeform_output.flatten()[:5]}")
            print(f"PyTorch Reference:    {torch_output.flatten()[:5]}")
    else:
        print("❌ Cannot compare outputs due to implementation failures")
    
    # print("\n" + "=" * 60)
    # print("⚡ Performance comparison:")
    
    # # 性能测试
    # if msdeform_success:
    #     msdeform_avg, msdeform_min = benchmark_function(msdeform_kernel, inputs, "MSDeformAttnFunction")
    
    # if torch_success:
    #     torch_avg, torch_min = benchmark_function(torch_kernel, inputs, "PyTorch Reference")
    
    # if msdeform_success and torch_success:
    #     speedup = torch_avg / msdeform_avg
    #     print(f"\n🏆 MSDeformAttnFunction is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch reference")
    
    print("\n" + "=" * 60)
    print("✨ Comparison completed!")

if __name__ == "__main__":
    main()
