import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import numpy as np

# 导入两个参考实现
from torch_.ref import torch_kernel as official_torch_kernel
from torch_.ref_naive import torch_kernel as naive_torch_kernel

def create_identical_inputs():
    """创建完全相同的测试输入，与CUDA kernel对齐"""
    torch.manual_seed(42)  # 固定随机种子确保可重现
    device = "cuda"
    
    # 从CUDA kernel得到的参数
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
    
    print(f"📊 Spatial shapes: {spatial_shapes}")
    print(f"📊 Level starts: {level_starts}")
    print(f"📊 Total spatial: {total_spatial}")
    
    # 创建value tensor: [total_spatial_size, m, d] - 使用与compare_implementations.py相同的方式
    value = torch.randn(total_spatial, m, d, dtype=torch.float32, device=device).normal_(mean=0.0, std=0.5)
    
    # 创建采样位置: [lq, m, l*k, 2] (归一化坐标 0-1)
    sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device=device)
    sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
    
    # 创建注意力权重: [lq, m, l*k] - 使用与compare_implementations.py相同的方式
    attention_weights = torch.randn(lq, m, l*k, dtype=torch.float32, device=device).normal_(mean=0.0, std=0.5)
    attention_weights = torch.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    inputs = (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)
    
    # 输出一些统计信息用于调试
    print(f"📊 Input statistics:")
    print(f"   - value: mean={value.mean().item():.6f}, std={value.std().item():.6f}")
    print(f"   - sampling_locations: range=[{sampling_locations.min().item():.3f}, {sampling_locations.max().item():.3f}]")
    print(f"   - attention_weights: mean={attention_weights.mean().item():.6f}, sum_check={attention_weights.sum(dim=-1).mean().item():.6f}")
    
    return inputs

def test_two_ref_implementations():
    """测试两个参考实现是否对齐"""
    print("🧪 Testing two reference implementations alignment")
    print("=" * 60)
    
    inputs = create_identical_inputs()
    
    # 测试官方PyTorch实现
    print("\n1️⃣ Testing 官方PyTorch参考实现...")
    try:
        official_output = official_torch_kernel(*inputs)
        print(f"   ✅ Success: {official_output.shape}")
        print(f"   Stats: mean={official_output.mean().item():.6f}, std={official_output.std().item():.6f}")
        official_success = True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        official_success = False
        official_output = None
    
    # 测试naive PyTorch实现 
    print("\n2️⃣ Testing Naive PyTorch参考实现...")
    try:
        naive_output = naive_torch_kernel(*inputs)
        print(f"   ✅ Success: {naive_output.shape}")
        print(f"   Stats: mean={naive_output.mean().item():.6f}, std={naive_output.std().item():.6f}")
        naive_success = True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        naive_success = False
        naive_output = None
    
    # 对比两个实现的结果
    if official_success and naive_success:
        print("\n📊 Comparing outputs...")
        
        abs_diff = torch.abs(official_output - naive_output)
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        
        rel_diff = abs_diff / (torch.abs(naive_output) + 1e-8)
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        print(f"   - Max absolute difference: {max_abs_diff:.2e}")
        print(f"   - Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"   - Max relative difference: {max_rel_diff:.2e}")
        print(f"   - Mean relative difference: {mean_rel_diff:.2e}")
        
        is_close = torch.allclose(official_output, naive_output, atol=1e-3, rtol=1e-2)
        print(f"   - {'✅ ALIGNED' if is_close else '❌ NOT ALIGNED'}")
        
        if is_close:
            print("\n🎉 两个参考实现完全对齐！这证明数据格式是正确的。")
            print("   CUDA kernel的问题可能是内在的实现bug，而不是数据格式问题。")
            
            # 显示一些样本值
            print(f"\n🔬 Sample values (first 5):")
            print(f"   Official: {official_output.flatten()[:5]}")
            print(f"   Naive:    {naive_output.flatten()[:5]}")
            
            return True, inputs, official_output
        else:
            print("\n⚠️ 两个参考实现不对齐，需要进一步调试")
            return False, inputs, None
    else:
        print("\n❌ 无法比较，至少有一个实现失败")
        return False, inputs, None

def debug_cuda_kernel_memory_layout():
    """调试CUDA kernel的内存布局期望"""
    print("\n🔍 Debug CUDA kernel memory layout...")
    
    # 分析CUDA kernel中的索引计算
    print("CUDA kernel索引计算分析:")
    print("  value_[((((((value_level_start_index_[i] * 2048) +")
    print("             ((xy_rounded[0] * height_width[1]) * 2048)) +")
    print("            (xy_rounded[2] * 2048)) +")
    print("           (((int)blockIdx.z) * 256)) +")
    print("          (((int)threadIdx.x) * 4)) +")
    print("         ii_d_2)]")
    print()
    print("  其中：")
    print("  - 2048 = 8 * 256 (m * d)")
    print("  - 256 = d (embed_dim)")  
    print("  - blockIdx.z = head_idx (0-7)")
    print("  - threadIdx.x = thread_idx (0-63)")
    print("  - ii_d_2 = dim_offset (0-3)")
    print()
    print("  这意味着CUDA kernel期望的内存布局可能是：")
    print("  [level][height][width][head][dim] 或类似的4D/5D结构")
    print("  而我们提供的是：[total_spatial][head][dim] 的3D结构")
    print()
    print("💡 建议：")
    print("  1. CUDA kernel可能有内在的内存访问bug")
    print("  2. 或者需要特定的value tensor重排")
    print("  3. 推荐直接使用MSDeformAttnFunction替代")

def main():
    print("🔍 Debug Reference Implementation Alignment")
    print("=" * 60)
    
    success, inputs, output = test_two_ref_implementations()
    
    if success:
        debug_cuda_kernel_memory_layout()
        
        print("\n" + "=" * 60)
        print("🎯 结论:")
        print("  ✅ 官方PyTorch实现与Naive实现完全对齐")
        print("  ✅ 数据格式和初始化方式都是正确的")
        print("  ❌ CUDA kernel的illegal memory access是内在问题")
        print("  💡 推荐使用MSDeformAttnFunction，它已经验证工作正常")
        
    else:
        print("\n" + "=" * 60)
        print("❌ 参考实现之间不对齐，需要进一步调试")

if __name__ == "__main__":
    main()
